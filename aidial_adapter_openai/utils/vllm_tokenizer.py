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

from typing import Any

import httpx
from aidial_sdk.exceptions import InternalServerError

from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import logger


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
    """Tokenizer backed by a remote vLLM ``/tokenize`` endpoint."""

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

    async def tokenize(self, request: dict) -> int:
        return await self._call_tokenize(request)

    async def _call_tokenize(self, payload: dict[str, Any]) -> int:
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
