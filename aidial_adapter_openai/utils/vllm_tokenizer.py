"""
Tokenizer that delegates token counting entirely to a vLLM uplink server.

vLLM exposes a ``/tokenize`` endpoint that accepts a list of messages
(including multi-modal content such as images and files encoded as base64)
and returns the total token count.  This tokenizer simply forwards the
already-transformed request payload to that endpoint and trusts the
returned count — no modality-specific accounting is done on the adapter
side.

The tokenize URL is derived programmatically from the upstream
chat-completions URL by replacing the ``/chat/completions`` path suffix
with ``/tokenize``.

This tokenizer does not do token counting for the *response*.
Instead, usage statistics are obtained from the upstream vLLM model
response (``usage`` block).
"""

import re
from typing import Any, Dict, List, Mapping

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


def derive_tokenize_url(upstream_endpoint: str) -> str:
    """Derive the vLLM ``/tokenize`` URL from the chat completions endpoint.

    vLLM exposes ``/tokenize`` at the server root, not under ``/v1/``.
    Therefore we strip the entire path suffix starting from ``/v1/``
    (or just ``/chat/completions`` when ``/v1/`` is absent)::

        https://host/v1/chat/completions                        -> https://host/tokenize
        https://host/chat/completions                           -> https://host/tokenize
        https://host/openai/deployments/<id>/chat/completions   -> https://host/openai/deployments/<id>/tokenize
    """

    if not upstream_endpoint.endswith("/chat/completions"):
        raise InternalServerError(
            f"Cannot derive vLLM tokenize URL from upstream endpoint: {upstream_endpoint!r}. "
            "Expected the endpoint to end with '/chat/completions'."
        )

    # Strip /v1/chat/completions (preferred) or /chat/completions
    url = re.sub(r"/v1/chat/completions$", "/tokenize", upstream_endpoint)
    if url == upstream_endpoint:
        # No /v1/ prefix — just replace /chat/completions
        url = re.sub(r"/chat/completions$", "/tokenize", upstream_endpoint)

    return url


class VllmTokenizer:
    """Tokenizer backed by a remote vLLM ``/tokenize`` endpoint.

    The tokenizer sends the **full** request payload (all messages, tools,
    etc.) to the vLLM server in a single call and returns the total token
    count reported by the server.  No per-message, per-modality, or
    per-attachment token counting is performed on the adapter side.

    This tokenizer **not** performs response token counting.
    The adapter forces usage reporting in upstream requests and extracts
    token counts from the model response.
    """

    model: str
    tokenize_url: str
    _api_key: str | None
    _http_client: httpx.AsyncClient

    def __init__(
        self,
        *,
        model: str,
        upstream_endpoint: str,
        request_headers: Mapping[str, str],
    ) -> None:
        self.model = model
        self.tokenize_url = derive_tokenize_url(upstream_endpoint)

        # Re-use the same API key that was forwarded to the upstream
        self._api_key = request_headers.get("api-key") or request_headers.get(
            "authorization", ""
        ).removeprefix("Bearer ").strip() or None

        self._http_client = get_http_client()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def tokenize_request(
        self, original_request: dict, messages: List[MultiModalMessage]
    ) -> int:
        """Count tokens for the full request (messages + tools/functions)
        via a single call to the vLLM tokenize endpoint.

        Each message is treated as an atomic unit; text, images, and files
        are sent together — no separate tokenization per modality.
        """

        raw_messages = [m.raw_message for m in messages]
        payload = self._build_tokenize_payload(original_request, raw_messages)
        return await self._call_tokenize(payload)

    async def truncate_prompt(
        self,
        original_request: dict,
        messages: List[MultiModalMessage],
        max_prompt_tokens: int,
    ) -> tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
        """Truncate messages to fit within *max_prompt_tokens*.

        The algorithm:
        1. Send the full message list to the vLLM tokenize endpoint.
        2. If the count exceeds the limit, remove the oldest non-system
           message and re-send the **entire** remaining list.
        3. Repeat until the list fits or only system messages remain.
        """

        system_indices: list[int] = []
        non_system_indices: list[int] = []
        for idx, msg in enumerate(messages):
            if msg.raw_message.get("role") == "system":
                system_indices.append(idx)
            else:
                non_system_indices.append(idx)

        kept_indices = set(range(len(messages)))

        def _current_messages() -> List[MultiModalMessage]:
            return [messages[i] for i in sorted(kept_indices)]

        # First: check if system messages alone already exceed the limit
        system_only = [messages[i] for i in system_indices]
        system_tokens = await self.tokenize_request(
            original_request, system_only
        )
        if system_tokens > max_prompt_tokens:
            raise TruncatePromptSystemError(max_prompt_tokens, system_tokens)

        # Try with all messages
        prompt_tokens = await self.tokenize_request(
            original_request, _current_messages()
        )

        if prompt_tokens <= max_prompt_tokens:
            return (
                _current_messages(),
                [],
                prompt_tokens,
            )

        # Iteratively remove oldest non-system messages (from the front)
        # until it fits.  Keep at least the last non-system message so we
        # can raise TruncatePromptSystemAndLastUserError if even that
        # doesn't fit.
        for remove_idx in non_system_indices[:-1]:
            kept_indices.discard(remove_idx)
            prompt_tokens = await self.tokenize_request(
                original_request, _current_messages()
            )
            if prompt_tokens <= max_prompt_tokens:
                discarded = sorted(set(range(len(messages))) - kept_indices)
                return (
                    _current_messages(),
                    discarded,
                    prompt_tokens,
                )

        # Only system + last non-system message left — if it still
        # doesn't fit, raise an error.
        if non_system_indices:
            last_non_system = non_system_indices[-1]
            kept_indices = set(system_indices) | {last_non_system}
            prompt_tokens = await self.tokenize_request(
                original_request, _current_messages()
            )
            if prompt_tokens > max_prompt_tokens:
                raise TruncatePromptSystemAndLastUserError(
                    max_prompt_tokens, prompt_tokens
                )

        discarded = sorted(set(range(len(messages))) - kept_indices)
        return (
            _current_messages(),
            discarded,
            prompt_tokens,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _build_tokenize_payload(
        self, original_request: dict, messages: List[dict]
    ) -> Dict[str, Any]:
        """Build the JSON body sent to the vLLM ``/tokenize`` endpoint."""

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # Forward tools/functions so the server accounts for their tokens
        if tools := original_request.get("tools"):
            payload["tools"] = tools
        if functions := original_request.get("functions"):
            payload["functions"] = functions

        return payload

    async def _call_tokenize(self, payload: Dict[str, Any]) -> int:
        """POST *payload* to the vLLM tokenize endpoint and return token count."""

        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        logger.debug(
            f"vLLM tokenize request to {self.tokenize_url}, "
            f"model={payload.get('model')}, "
            f"messages_count={len(payload.get('messages', []))}"
        )

        try:
            response = await self._http_client.post(
                self.tokenize_url,
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
                f"Failed to reach vLLM tokenize endpoint at {self.tokenize_url}: {exc}"
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

