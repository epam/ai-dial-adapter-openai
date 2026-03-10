from unittest.mock import AsyncMock

import httpx
import pytest
from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.vllm_tokenizer import (
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
    async def test_tokenize_request_forwards_structured_content_unchanged(self):
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(20)
        tokenizer._http_client = mock_client

        structured_content = [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,..."},
            },
        ]

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": structured_content}
            ),
        ]

        await tokenizer.tokenize_request({"model": "my-vllm-model"}, messages)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["messages"][1]["content"] == structured_content

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
    async def test_no_system_message_truncation_succeeds(self):
        """No system message — just user/assistant turns. Oldest messages
        are dropped until the remainder fits."""
        # Call 1: full list (4 msgs) → 40 (exceeds 20)
        # Call 2: after dropping u1 → 30 (still exceeds)
        # Call 3: after dropping assistant reply → 15 (fits)
        tokenizer = _make_mock_tokenizer([40, 30, 15])

        messages = [
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "a1"}
            ),
            MultiModalMessage(raw_message={"role": "user", "content": "u2"}),
            MultiModalMessage(
                raw_message={"role": "assistant", "content": "a2"}
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 20
        )

        assert sorted(discarded) == [0, 1]
        assert used == 15
        assert len(truncated) == 2
        assert truncated[0].raw_message["content"] == "u2"
        assert truncated[1].raw_message["content"] == "a2"

    @pytest.mark.asyncio
    async def test_no_system_message_last_message_too_big(self):
        """No system message and even the single remaining last message
        exceeds the budget — raises TruncatePromptSystemAndLastUserError."""
        # Call 1: full list [u0, u1] → 50 (exceeds 10)
        # Call 2: loop drops u0, tokenize [u1] → 50 (still exceeds, loop ends)
        # No system messages → skip system-only check, raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([50, 50])

        messages = [
            MultiModalMessage(raw_message={"role": "user", "content": "old"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "huge last message"}
            ),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_raises_system_error(self):
        """System messages alone exceed the budget."""
        # Call 1: full list (system-only) → 50 (exceeds budget of 10)
        # No loop iterations (non_system_indices is empty)
        # Call 2: system-only → 50 => raise TruncatePromptSystemError
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
        # Call 1: full list [sys, user] → 50 (exceeds 10)
        # No loop iterations (non_system_indices[:-1] is empty)
        # last_measured_tokens = 50 (from call 1, already system+last)
        # Call 2: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([50, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "huge"}),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await tokenizer.truncate_prompt({}, messages, 10)

    @pytest.mark.asyncio
    async def test_structured_user_content_still_raises_last_user_error(self):
        """truncate_prompt works with message boundaries; structured content
        does not change SystemAndLastUser behavior."""
        # Call 1: full list [sys, user] → 200 (exceeds 100)
        # No loop iterations (non_system_indices[:-1] is empty)
        # last_measured_tokens = 200 (from call 1, already system+last)
        # Call 2: system-only → 5 (fits) => raise SystemAndLastUser
        tokenizer = _make_mock_tokenizer([200, 5])

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
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
    async def test_structured_user_content_kept_when_fits(self):
        """Structured user content is kept intact when truncation keeps that message."""
        # Call 1: full list → 80 (exceeds 60)
        # Call 2: after removing plain (system + structured) → 55 (fits)
        tokenizer = _make_mock_tokenizer([80, 55])

        structured_content = [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,..."},
            },
        ]

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": "old msg"}
            ),
            MultiModalMessage(
                raw_message={"role": "user", "content": structured_content},
            ),
        ]

        truncated, discarded, used = await tokenizer.truncate_prompt(
            {}, messages, 60
        )

        assert discarded == [1]
        assert used == 55
        assert len(truncated) == 2
        assert truncated[0].raw_message["role"] == "system"
        assert truncated[1].raw_message["content"] == structured_content

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

        structured_content = [
            {"type": "text", "text": "describe this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/jpeg;base64,..."},
            },
        ]

        messages = [
            MultiModalMessage(raw_message={"role": "system", "content": "sys"}),
            MultiModalMessage(raw_message={"role": "user", "content": "u1"}),
            MultiModalMessage(
                raw_message={"role": "user", "content": structured_content},
            ),
        ]

        await tokenizer.truncate_prompt({}, messages, 20)

        # Call 1: full list (3 messages)
        assert len(call_payloads[0]) == 3

        # Call 2: after dropping u1 → system + structured user message (2 messages)
        assert len(call_payloads[1]) == 2
        assert call_payloads[1][0]["content"] == "sys"
        assert call_payloads[1][1]["content"] == structured_content


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
