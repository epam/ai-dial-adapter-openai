"""
Tokenizer that delegates token counting entirely to a vLLM uplink server.

vLLM exposes a ``/tokenize`` endpoint that accepts a list of messages
(including multi-modal content such as images and files encoded as base64)
and returns the total token count.  This tokenizer simply forwards the
already-transformed request payload to that endpoint and trusts the
returned count — no modality-specific accounting is done on the adapter
side.

The tokenizing endpoint URL is derived programmatically from the upstream
chat-completions URL by replacing the ``v1/chat/completions`` path suffix
with ``/tokenize``.
"""

from typing import Any, List, Set

import httpx
from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
)

# NOTE: vLLM truncation uses a simple linear strategy; we intentionally
# avoid binary search to keep implementation straightforward.


def derive_tokenize_url(upstream_endpoint: str) -> str:
    """Derive the vLLM ``/tokenize`` URL from the chat completions endpoint.

    Only the standard vLLM endpoint shape is supported::

        https://host/v1/chat/completions  ->  https://host/tokenize
    """

    if not upstream_endpoint.endswith("/v1/chat/completions"):
        raise InternalServerError(
            f"Cannot derive vLLM tokenize URL from upstream endpoint: {upstream_endpoint!r}. "
            "Expected the endpoint to end with '/v1/chat/completions'."
        )

    return upstream_endpoint.removesuffix("/v1/chat/completions") + "/tokenize"


class VllmTokenizer:
    """Tokenizer backed by a remote vLLM ``/tokenize`` endpoint.

    The tokenizer sends the **full** request payload (all messages, tools,
    etc.) to the vLLM server in a single call and returns the total token
    count reported by the server.  No per-message, per-modality, or
    per-attachment token counting is performed on the adapter side.
    """

    _tokenize_url: str
    _extra_headers: dict[str, str]
    _http_client: httpx.AsyncClient

    def __init__(
        self,
        *,
        upstream_endpoint: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self._tokenize_url = derive_tokenize_url(upstream_endpoint)
        self._extra_headers = extra_headers or {}

        self._http_client = get_http_client()

    async def tokenize_request(
        self, original_request: dict, messages: List[MultiModalMessage]
    ) -> int:
        """Count tokens for the full request (messages + tools/functions)
        via a single call to the vLLM tokenize endpoint.

        Each message is treated as an atomic unit; text, images, and files
        are sent together — no separate tokenization per modality.
        """

        raw_messages = [m.raw_message for m in messages]
        payload = {**original_request, "messages": raw_messages}
        return await self._call_tokenize(payload)

    async def truncate_prompt(
        self,
        original_request: dict,
        messages: List[MultiModalMessage],
        max_prompt_tokens: int,
    ) -> tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
        """Truncate messages to fit within *max_prompt_tokens*.

        vLLM token counting is delegated to the upstream ``/tokenize``
        endpoint.

        Behavior:
        - Try the full payload first; if it fits, return immediately.
        - Otherwise, remove the oldest non-system messages one-by-one
          (the last non-system message is never removed).
        - If a removed message is an assistant message with ``tool_calls``,
          also remove all the following ``tool`` messages and the next
          ``assistant`` message that follows the tool chain.
        - If even ``system + last_non_system`` doesn't fit, raise
          ``TruncatePromptSystemAndLastUserError`` (or
          ``TruncatePromptSystemError`` when system tokens alone exceed
          the limit).
        - If there are no non-system messages, the prompt consists only of
          system messages; if they exceed the limit,
          raise ``TruncatePromptSystemError``.
        """

        all_indices: Set[int] = set(range(len(messages)))

        def _collect(indices: Set[int]) -> List[MultiModalMessage]:
            return [messages[i] for i in sorted(indices)]

        # Fast path
        prompt_tokens = await self.tokenize_request(
            original_request, _collect(all_indices)
        )
        if prompt_tokens <= max_prompt_tokens:
            return _collect(all_indices), [], prompt_tokens

        system_indices: list[int] = []
        non_system_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.raw_message.get("role") == "system":
                system_indices.append(idx)
            else:
                non_system_indices.append(idx)

        system_set: Set[int] = set(system_indices)
        kept: Set[int] = set(all_indices)

        def _cascade_remove_tool_replies(start_idx: int) -> None:
            """Remove consecutive tool replies following *start_idx* and the next assistant."""
            i = start_idx + 1
            while i < len(messages):
                if i not in kept:
                    i += 1
                    continue
                role = messages[i].raw_message.get("role")
                if role == "tool":
                    kept.discard(i)
                    i += 1
                    continue
                if role == "assistant":
                    kept.discard(i)
                    break
                # If it's a user/system/etc. stop cascading.
                break

        # Remove the oldest non-system messages but keep the last non-system.
        # Track the token count from the most recent tokenize call so we don't
        # re-tokenize the same set in the final check below.
        last_measured_tokens = prompt_tokens
        for idx in non_system_indices[:-1]:
            if idx not in kept:
                continue

            raw = messages[idx].raw_message
            kept.discard(idx)

            # If we remove an assistant with tool_calls, also remove tool
            # replies until the next assistant.
            if raw.get("role") == "assistant" and raw.get("tool_calls"):
                _cascade_remove_tool_replies(idx)

            last_measured_tokens = await self.tokenize_request(
                original_request, _collect(kept)
            )
            if last_measured_tokens <= max_prompt_tokens:
                discarded = sorted(all_indices - kept)
                return _collect(kept), discarded, last_measured_tokens

        # All droppable messages have been removed; `kept` now holds
        # system messages + last non-system message (or just system messages
        # if there were no non-system messages). `last_measured_tokens` is
        # already the token count for the current `kept` set.
        if non_system_indices:
            # last_measured_tokens == tokenize(system + last non-system).
            # Check whether system alone is the bottleneck — only meaningful
            # when there are system messages to check.
            if system_set:
                system_tokens = await self.tokenize_request(
                    original_request, _collect(system_set)
                )
                if system_tokens > max_prompt_tokens:
                    raise TruncatePromptSystemError(
                        max_prompt_tokens, system_tokens
                    )

            raise TruncatePromptSystemAndLastUserError(
                max_prompt_tokens, last_measured_tokens
            )
        else:
            # No non-system messages — last_measured_tokens == tokenize(system only).
            raise TruncatePromptSystemError(
                max_prompt_tokens, last_measured_tokens
            )

    async def _call_tokenize(self, payload: dict[str, Any]) -> int:
        """POST *payload* to the vLLM tokenize endpoint and return token count."""

        headers: dict[str, str] = {"Content-Type": "application/json"}
        # vLLM /tokenize does not require authorization.
        if self._extra_headers:
            headers.update(self._extra_headers)

        logger.debug(
            f"vLLM tokenize request to {self._tokenize_url}, "
            f"model={payload.get('model')}, "
            f"messages_count={len(payload.get('messages', []))}"
        )

        try:
            response = await self._http_client.post(
                self._tokenize_url,
                json=payload,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"vLLM tokenize endpoint returned {exc.response.status_code}: "
                f"{exc.response.text}"
            )
            raise InternalServerError(
                f"vLLM tokenize endpoint error (HTTP {exc.response.status_code}): "
                f"{exc.response.text}"
            )
        except httpx.HTTPError as exc:
            logger.error(f"vLLM tokenize request failed: {exc}")
            raise InternalServerError(
                f"Failed to reach vLLM tokenize endpoint at {self._tokenize_url}: {exc}"
            )

        data = response.json()

        # vLLM /tokenize response schema:
        #   {"count": <int>, "max_model_len": <int>, "tokens": [...]}
        # We use "count" when available; otherwise fall back to len(tokens).
        if "count" in data:
            token_count = int(data["count"])
        elif "tokens" in data:
            token_count = len(data["tokens"])
        else:
            logger.error(f"Unexpected vLLM tokenize response: {data}")
            raise InternalServerError(
                "vLLM tokenize response does not contain 'count' or 'tokens' field."
            )

        logger.debug(f"vLLM tokenize result: {token_count} tokens")
        return token_count
