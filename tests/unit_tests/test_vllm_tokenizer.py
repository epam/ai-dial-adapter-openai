from unittest.mock import AsyncMock

import httpx
import pytest
from aidial_sdk.exceptions import InternalServerError

from aidial_adapter_openai.chat_completions.vllm.tokenizer import (
    VllmTokenizer,
    derive_tokenize_url,
)


class TestDeriveTokenizeUrl:
    def test_v1_chat_completions(self):
        url = "https://vllm.example.com/v1/chat/completions"
        assert derive_tokenize_url(url) == "https://vllm.example.com/tokenize"

    def test_with_port(self):
        url = "http://localhost:17834/v1/chat/completions"
        assert derive_tokenize_url(url) == "http://localhost:17834/tokenize"

    def test_plain_chat_completions_raises(self):
        with pytest.raises(InternalServerError):
            derive_tokenize_url("https://vllm.example.com/chat/completions")

    def test_non_chat_completions_raises(self):
        with pytest.raises(InternalServerError):
            derive_tokenize_url("https://vllm.example.com/v1/completions")

    def test_arbitrary_path_raises(self):
        with pytest.raises(InternalServerError):
            derive_tokenize_url(
                "https://host.com/openai/deployments/my-model/chat/completions"
            )


_UPSTREAM = "https://vllm.example.com/v1/chat/completions"


def _make_tokenizer() -> VllmTokenizer:
    return VllmTokenizer(
        upstream_endpoint=_UPSTREAM,
    )


def _mock_response(token_count: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "count": token_count,
            "tokens": list(range(token_count)),
        },
        request=httpx.Request("POST", _UPSTREAM),
    )


class TestVllmTokenizerTokenize:
    @pytest.mark.asyncio
    async def test_returns_count_from_response(self):
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(42)
        tokenizer._http_client = mock_client

        result = await tokenizer.tokenize({"model": "m", "messages": []})
        assert result == 42

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "Authorization" not in headers

    @pytest.mark.asyncio
    async def test_falls_back_to_tokens_length(self):
        tokenizer = _make_tokenizer()

        resp = httpx.Response(
            200,
            json={"tokens": [1, 2, 3]},
            request=httpx.Request("POST", _UPSTREAM),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = resp
        tokenizer._http_client = mock_client

        result = await tokenizer.tokenize({"model": "m", "messages": []})
        assert result == 3

    @pytest.mark.asyncio
    async def test_raises_on_missing_fields(self):
        tokenizer = _make_tokenizer()

        resp = httpx.Response(
            200,
            json={"something": "else"},
            request=httpx.Request("POST", _UPSTREAM),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = resp
        tokenizer._http_client = mock_client

        with pytest.raises(InternalServerError):
            await tokenizer.tokenize({"model": "m", "messages": []})

    @pytest.mark.asyncio
    async def test_raises_on_http_error(self):
        tokenizer = _make_tokenizer()

        resp = httpx.Response(
            500,
            text="Internal Server Error",
            request=httpx.Request("POST", _UPSTREAM),
        )
        mock_client = AsyncMock()
        mock_client.post.return_value = resp
        tokenizer._http_client = mock_client

        with pytest.raises(InternalServerError, match="HTTP 500"):
            await tokenizer.tokenize({"model": "m", "messages": []})


class TestVllmTokenizerForwardingAndHeaders:
    @staticmethod
    def _build_request() -> dict:
        return {
            "model": "my-vllm-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "i" * 10},
                        },
                    ],
                },
            ],
            "tools": [
                {"type": "function", "function": {"name": "f"}},
            ],
        }

    @pytest.mark.asyncio
    async def test_tokenize_forwards_full_request_without_extra_headers(self):
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(50)
        tokenizer._http_client = mock_client

        request = self._build_request()
        result = await tokenizer.tokenize(request)

        assert result == 50
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")

        assert payload == request
        assert headers["Content-Type"] == "application/json"
        assert "Authorization" not in headers
        assert "x-user-id" not in headers

    @pytest.mark.asyncio
    async def test_tokenize_forwards_full_request_with_extra_headers(self):
        tokenizer = VllmTokenizer(
            upstream_endpoint=_UPSTREAM,
            extra_headers={"x-user-id": "abc123", "x-custom": "value"},
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(51)
        tokenizer._http_client = mock_client

        request = self._build_request()
        result = await tokenizer.tokenize(request)

        assert result == 51
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")

        assert payload == request
        assert headers["Content-Type"] == "application/json"
        assert headers["x-user-id"] == "abc123"
        assert headers["x-custom"] == "value"
