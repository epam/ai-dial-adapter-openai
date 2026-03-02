"""Unit tests for VllmTokenizer and vLLM-based prompt truncation."""

from unittest.mock import AsyncMock

import httpx
import pytest
from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource
from aidial_adapter_openai.utils.vllm_tokenizer import (
    VllmTokenizer,
    derive_tokenize_url,
)

# ---------------------------------------------------------------
# derive_tokenize_url
# ---------------------------------------------------------------


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


# ---------------------------------------------------------------
# VllmTokenizer helpers
# ---------------------------------------------------------------

_UPSTREAM = "https://vllm.example.com/v1/chat/completions"


def _make_tokenizer() -> VllmTokenizer:
    return VllmTokenizer(
        model="my-vllm-model",
        upstream_endpoint=_UPSTREAM,
    )


def _mock_response(token_count: int) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "count": token_count,
            "max_model_len": 4096,
            "tokens": list(range(token_count)),
        },
        request=httpx.Request("POST", _UPSTREAM),
    )


# ---------------------------------------------------------------
# VllmTokenizer._call_tokenize
# ---------------------------------------------------------------


class TestVllmTokenizerCallTokenize:
    @pytest.mark.asyncio
    async def test_returns_count_from_response(self):
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(42)
        tokenizer._http_client = mock_client

        result = await tokenizer._call_tokenize({"model": "m", "messages": []})
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

        result = await tokenizer._call_tokenize({"model": "m", "messages": []})
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
            await tokenizer._call_tokenize({"model": "m", "messages": []})

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
            await tokenizer._call_tokenize({"model": "m", "messages": []})


# ---------------------------------------------------------------
# VllmTokenizer.tokenize_request
# ---------------------------------------------------------------


class TestVllmTokenizerPublicApi:
    @pytest.mark.asyncio
    async def test_tokenize_request_sends_full_message_list_and_tools(self):
        """tokenize_request must send ALL messages in a single call,
        along with tools/functions."""
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(50)
        tokenizer._http_client = mock_client

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "hi"}),
        ]
        original_request = {
            "model": "my-vllm-model",
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }

        result = await tokenizer.tokenize_request(original_request, messages)
        assert result == 50

        # Verify a single call was made with both messages
        assert mock_client.post.call_count == 1
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "my-vllm-model"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}
        assert payload["tools"] == original_request["tools"]

    @pytest.mark.asyncio
    async def test_tokenize_request_with_empty_messages(self):
        """tokenize_request([]) sends an empty list — used for overhead."""
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(3)
        tokenizer._http_client = mock_client

        result = await tokenizer.tokenize_request({"model": "m"}, [])
        assert result == 3

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["messages"] == []


# ---------------------------------------------------------------
# VllmTokenizer.truncate_prompt  (full-list tokenization)
# ---------------------------------------------------------------


def _make_mock_tokenizer(responses: list[int]) -> VllmTokenizer:
    """Create a VllmTokenizer where successive tokenize_request calls
    return counts from *responses* in order."""
    tokenizer = _make_tokenizer()
    call_index = {"idx": 0}
    call_log: list[int] = []  # message counts per call

    async def mock_tokenize_request(original_request, messages):
        idx = call_index["idx"]
        call_index["idx"] += 1
        call_log.append(len(messages))
        if idx < len(responses):
            return responses[idx]
        return responses[-1]

    tokenizer.tokenize_request = mock_tokenize_request  # type: ignore[assignment]
    tokenizer._mock_call_log = call_log  # type: ignore[attr-defined]
    return tokenizer


# Removed: _make_counting_tokenizer (no longer used)


class TestVllmTruncatePrompt:
    @pytest.mark.asyncio
    async def test_fits_without_truncation(self):
        """All messages fit — no truncation needed."""
        # Call 1: full list → 20
        tokenizer = _make_mock_tokenizer([20])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "hi"}),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 30
        )

        assert discarded == []
        assert used == 20
        assert len(truncated) == 2

    @pytest.mark.asyncio
    async def test_drops_oldest_non_system_message(self):
        """Three messages: system + 2 user.  Full list is too big, so
        oldest user message is dropped and the full remaining list is
        re-tokenized."""
        # Call 1: full list (3 msgs) → 30 (exceeds 25)
        # Call 2: after dropping oldest group → 18 (fits)
        tokenizer = _make_mock_tokenizer([30, 18])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "old user"}
            ),
            MultiModalMessage(
                raw_message={"role": "user", "content": "new user"}
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 25
        )

        assert discarded == [1]
        assert used == 18
        assert len(truncated) == 2
        assert truncated[0].raw_message["content"] == "sys"
        assert truncated[1].raw_message["content"] == "new user"

    @pytest.mark.asyncio
    async def test_drops_multiple_messages(self):
        """Four messages: system + 3 user.  Need to drop 2 oldest."""
        # Call 1: full list (4 msgs) → 40 (exceeds 15)
        # Call 2: after dropping group1 → 30 (still exceeds)
        # Call 3: after dropping group2 → 12 (fits)
        tokenizer = _make_mock_tokenizer([40, 30, 12])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u2"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u3"}),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 15
        )

        assert sorted(discarded) == [1, 2]
        assert used == 12
        assert len(truncated) == 2
        assert truncated[0].raw_message["content"] == "sys"
        assert truncated[1].raw_message["content"] == "u3"

    @pytest.mark.asyncio
    async def test_raises_system_error(self):
        """System messages alone exceed the budget."""
        # Call 1: full list (system-only) → 50 (exceeds budget of 10)
        # Call 2: system-only confirmation → 50
        tokenizer = _make_mock_tokenizer([50, 50])

        messages = [
            MultiModalMessage(
                raw_message={"role": "system", "content": "long"}
            ),
        ]

        with pytest.raises(TruncatePromptSystemError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_raises_system_and_last_user_error(self):
        """System + last user message exceeds the budget."""
        # Call 1: full list → 50 (exceeds 10)
        # Call 2: system + last user → 50 (still exceeds)
        # Call 3: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([50, 50, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "huge"}),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_multimodal_messages_sent_as_whole(self):
        """Messages with images/files are sent as atomic units.
        The tokenize endpoint sees the full content (including base64)."""
        # Call 1: full list → 200 (exceeds 100)
        # Call 2: system + last multimodal → 200 (still exceeds)
        # Call 3: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([200, 200, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                images=[
                    ImageResource(
                        width=100,
                        height=100,
                        detail="low",
                        image=Resource(type="image/jpeg", data=b"..."),
                    )
                ],
                raw_message={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,..."},
                        },
                    ],
                },
            ),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 100)

    @pytest.mark.asyncio
    async def test_multimodal_message_kept_when_fits(self):
        """Multimodal message fits after dropping older plain messages."""
        # Call 1: full list → 80 (exceeds 60)
        # Call 2: after removing plain (system + multimodal) → 55 (fits)
        tokenizer = _make_mock_tokenizer([80, 55])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "old msg"}
            ),
            MultiModalMessage(
                images=[
                    ImageResource(
                        width=100,
                        height=100,
                        detail="low",
                        image=Resource(type="image/jpeg", data=b"..."),
                    )
                ],
                raw_message={
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe this"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64,..."},
                        },
                    ],
                },
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 60
        )

        assert discarded == [1]
        assert used == 55
        assert len(truncated) == 2
        assert truncated[0].raw_message["role"] == "system"
        # The multimodal message (with image) is kept as-is
        assert isinstance(truncated[1].raw_message["content"], list)

    @pytest.mark.asyncio
    async def test_tokenize_called_with_full_list_each_iteration(self):
        """Verify that each truncation step re-sends the full remaining
        message list (not individual messages)."""
        tokenizer = _make_tokenizer()

        call_payloads = []

        async def capturing_tokenize_request(original_request, messages):
            raw = [m.raw_message for m in messages]
            call_payloads.append(raw)
            # Simulate: full=30, after drop=15
            counts = [30, 15]
            idx = len(call_payloads) - 1
            return counts[idx] if idx < len(counts) else 15

        tokenizer.tokenize_request = capturing_tokenize_request  # type: ignore[assignment]

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u2"}),
        ]

        await tokenizer.truncate_prompt({}, messages, 20)

        # Call 1: full list (3 messages)
        assert len(call_payloads[0]) == 3

        # Call 2: after dropping u1 → system + u2 (2 messages)
        assert len(call_payloads[1]) == 2
        assert call_payloads[1][0]["content"] == "sys"
        assert call_payloads[1][1]["content"] == "u2"


# ---------------------------------------------------------------
# Tool-call cascade removal
# ---------------------------------------------------------------


class TestVllmToolCallCascade:
    @pytest.mark.asyncio
    async def test_assistant_tool_calls_cascade_removes_tool_messages(self):
        """When an assistant message with tool_calls is dropped, the adapter
        must also drop subsequent tool messages until the next assistant."""

        # Call 1: full (6 msgs) → 100 (exceeds)
        # Call 2: after dropping assistant+tool replies group → 12 (fits)
        tokenizer = _make_mock_tokenizer([100, 12])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                }
            ),
            # tool replies (should be removed with cascade)
            MultiModalMessage(
                raw_message={
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "r1",
                }
            ),
            MultiModalMessage(
                raw_message={
                    "role": "tool",
                    "tool_call_id": "call_1",
                    "content": "r2",
                }
            ),
            # next assistant breaks cascade
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "next"}
            ),
            MultiModalMessage(
                raw_message={"role": "user", "content": "follow-up"}
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 20
        )

        assert sorted(discarded) == [1, 2, 3, 4]
        assert used == 12
        assert truncated[-1].raw_message["content"] == "follow-up"

    @pytest.mark.asyncio
    async def test_non_tool_call_assistant_no_cascade(self):
        """A plain assistant message (no tool_calls) does not cascade."""
        tokenizer = _make_mock_tokenizer([60, 20])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "reply"}
            ),
            MultiModalMessage(
                raw_message={"role": "tool", "content": "orphan"}
            ),
            MultiModalMessage(raw_message={"role": "user", "content": "new_q"}),
        ]

        _, discarded, used = await tokenizer.truncate_prompt({}, messages, 25)

        # Only assistant dropped; tool message stays (no cascade).
        assert sorted(discarded) == [1]
        assert used == 20


# Removed: Binary search for long histories and TestVllmBinarySearch (no longer used)


# ---------------------------------------------------------------
# Extra headers (VLLM_HEADERS_TO_PROXY support)
# ---------------------------------------------------------------


class TestVllmExtraHeaders:
    @pytest.mark.asyncio
    async def test_extra_headers_included_in_tokenize_request(self):
        """Extra headers (from VLLM_HEADERS_TO_PROXY) are sent with tokenize calls."""
        tokenizer = VllmTokenizer(
            model="my-vllm-model",
            upstream_endpoint=_UPSTREAM,
            extra_headers={"x-user-id": "abc123", "x-custom": "value"},
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(10)
        tokenizer._http_client = mock_client

        await tokenizer._call_tokenize({"model": "m", "messages": []})

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert headers["x-user-id"] == "abc123"
        assert headers["x-custom"] == "value"
        assert headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_no_extra_headers_when_not_configured(self):
        """Without extra_headers, only standard headers are sent."""
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(10)
        tokenizer._http_client = mock_client

        await tokenizer._call_tokenize({"model": "m", "messages": []})

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        assert "x-user-id" not in headers
        assert "Content-Type" in headers

    @pytest.mark.asyncio
    async def test_extra_headers_empty_dict_is_noop(self):
        """Passing an empty dict for extra_headers is the same as None."""
        tokenizer = VllmTokenizer(
            model="my-vllm-model",
            upstream_endpoint=_UPSTREAM,
            extra_headers={},
        )

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(10)
        tokenizer._http_client = mock_client

        await tokenizer._call_tokenize({"model": "m", "messages": []})

        call_args = mock_client.post.call_args
        headers = call_args.kwargs.get("headers") or call_args[1].get("headers")
        # Only Content-Type should be present
        assert "x-user-id" not in headers
        assert headers["Content-Type"] == "application/json"
