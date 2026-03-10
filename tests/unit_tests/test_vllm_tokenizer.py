from unittest.mock import AsyncMock

import httpx
import pytest
from aidial_sdk.exceptions import (
    InternalServerError,
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.truncate_prompt import (
    truncate_prompt,
)
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
# Plain repeated chars keep synthetic payload lengths obvious in tests.
_TEST_IMAGE_URL = "i" * 10


def _make_tokenizer() -> VllmTokenizer:
    return VllmTokenizer(
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
    async def test_tokenize_sends_full_message_list_and_tools(self):
        """tokenize must send ALL messages in a single call,
        along with tools/functions."""
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(50)
        tokenizer._http_client = mock_client

        request = {
            "model": "my-vllm-model",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
            ],
            "tools": [{"type": "function", "function": {"name": "f"}}],
        }

        result = await tokenizer.tokenize(request)
        assert result == 50

        # Verify a single call was made with both messages
        assert mock_client.post.call_count == 1
        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["model"] == "my-vllm-model"
        assert len(payload["messages"]) == 2
        assert payload["messages"][0] == {"role": "system", "content": "sys"}
        assert payload["messages"][1] == {"role": "user", "content": "hi"}
        assert payload["tools"] == request["tools"]

    @pytest.mark.asyncio
    async def test_tokenize_forwards_structured_content_unchanged(self):
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(20)
        tokenizer._http_client = mock_client

        structured_message = _mi(text="describe this")

        request = {
            "model": "my-vllm-model",
            "messages": [
                {"role": "system", "content": "sys"},
                structured_message.raw_message,
            ],
        }

        await tokenizer.tokenize(request)

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert (
            payload["messages"][1]["content"]
            == structured_message.raw_message["content"]
        )

    @pytest.mark.asyncio
    async def test_tokenize_with_empty_messages(self):
        """tokenize with empty messages list."""
        tokenizer = _make_tokenizer()

        mock_client = AsyncMock()
        mock_client.post.return_value = _mock_response(3)
        tokenizer._http_client = mock_client

        result = await tokenizer.tokenize({"model": "m", "messages": []})
        assert result == 3

        call_args = mock_client.post.call_args
        payload = call_args.kwargs.get("json") or call_args[1].get("json")
        assert payload["messages"] == []


def _mm(role: str, content, **extra) -> MultiModalMessage:
    raw_message = {"role": role, "content": content}
    raw_message.update(extra)
    return MultiModalMessage(raw_message=raw_message)


def _mi(*, role: str = "user", text: str, image_url: str = _TEST_IMAGE_URL):
    return _mm(
        role,
        [
            {"type": "text", "text": text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    )


def _ma(
    *,
    content=None,
    function_name: str = "fn",
    arguments: str = "{}",
    call_id: str = "call_1",
):
    return _mm(
        "assistant",
        content,
        tool_calls=[
            {
                "id": call_id,
                "type": "function",
                "function": {"name": function_name, "arguments": arguments},
            }
        ],
    )


def _content_len(content) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_len(item) for item in content)
    if isinstance(content, dict):
        total = 0
        text = content.get("text")
        if isinstance(text, str):
            total += len(text)

        image_url = content.get("image_url")
        if isinstance(image_url, str):
            total += len(image_url)
        elif isinstance(image_url, dict):
            url = image_url.get("url")
            if isinstance(url, str):
                total += len(url)

        return total
    return 0


def _tool_calls_len(tool_calls) -> int:
    if not isinstance(tool_calls, list):
        return 0

    total = 0
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function")
        if not isinstance(function, dict):
            continue

        name = function.get("name")
        if isinstance(name, str):
            total += len(name)

        arguments = function.get("arguments")
        if isinstance(arguments, str):
            total += len(arguments)

    return total


def _message_len(message: MultiModalMessage) -> int:
    raw = message.raw_message
    total = _content_len(raw.get("content"))

    # Count agentic metadata to emulate non-text overhead in tests.
    total += _tool_calls_len(raw.get("tool_calls"))

    tool_call_id = raw.get("tool_call_id")
    if isinstance(tool_call_id, str):
        total += len(tool_call_id)

    return total


def _messages_char_count(messages: list[MultiModalMessage]) -> int:
    return sum(_message_len(m) for m in messages)


def _make_length_based_tokenizer(
    *, call_payloads: list[list[dict]] | None = None
) -> VllmTokenizer:
    """Create a tokenizer mock that deterministically counts message content chars.

    The mock replaces :meth:`VllmTokenizer.tokenize` so that the standalone
    :func:`truncate_prompt` function can be tested without real HTTP calls.
    Token count is computed as the total character length of message content
    in the request's ``messages`` list.
    """
    tokenizer = _make_tokenizer()

    def _char_count_from_request(request: dict) -> int:
        raw_messages = request.get("messages", [])
        # Wrap in MultiModalMessage to reuse the existing _message_len helper.
        wrapped = [MultiModalMessage(raw_message=m) for m in raw_messages]
        if call_payloads is not None:
            call_payloads.append(raw_messages)
        return _messages_char_count(wrapped)

    tokenizer.tokenize = AsyncMock(side_effect=_char_count_from_request)  # type: ignore[assignment]
    return tokenizer


class TestVllmTruncatePrompt:
    @pytest.mark.asyncio
    async def test_fits_without_truncation(self):
        """All messages fit — no truncation needed."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 5),
            _mm("user", "u" * 4),
        ]

        truncated, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count(messages),
        )

        assert discarded == []
        assert used == _messages_char_count(messages)
        assert truncated == messages

    @pytest.mark.asyncio
    async def test_drops_oldest_non_system_message(self):
        """Three messages: system + 2 user. Oldest user is dropped."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 3),
            _mm("user", "o" * 10),
            _mm("user", "n" * 8),
        ]

        truncated, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count([messages[0], messages[2]]),
        )

        assert discarded == [1]
        assert used == _messages_char_count(truncated)
        assert truncated == [messages[0], messages[2]]

    @pytest.mark.asyncio
    async def test_drops_multiple_messages(self):
        """Four messages: system + 3 user. Need to drop 2 oldest."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 3),
            _mm("user", "u" * 9),
            _mm("user", "v" * 8),
            _mm("user", "w" * 7),
        ]

        truncated, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count([messages[0], messages[3]]),
        )

        assert sorted(discarded) == [1, 2]
        assert used == _messages_char_count(truncated)
        assert truncated == [messages[0], messages[3]]

    @pytest.mark.asyncio
    async def test_no_system_message_truncation_succeeds(self):
        """No system message — oldest messages are dropped until fit."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("user", "u" * 9),
            _mm("assistant", "a" * 8),
            _mm("user", "v" * 6),
            _mm("assistant", "b" * 5),
        ]

        truncated, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count([messages[2], messages[3]]),
        )

        assert sorted(discarded) == [0, 1]
        assert used == _messages_char_count(truncated)
        assert truncated == [messages[2], messages[3]]

    @pytest.mark.asyncio
    async def test_no_system_message_last_message_too_big(self):
        """No system message and last remaining message still exceeds budget."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("user", "o" * 3),
            _mm("user", "h" * 11),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await truncate_prompt(
                tokenizer,
                {},
                messages,
                lambda m: m.raw_message,
                _messages_char_count([messages[0]]),
            )

    @pytest.mark.asyncio
    async def test_raises_system_error(self):
        """System messages alone exceed the budget."""
        tokenizer = _make_length_based_tokenizer()

        sys_msg = _mm("system", "s" * 11)
        messages = [sys_msg]

        with pytest.raises(TruncatePromptSystemError):
            await truncate_prompt(
                tokenizer,
                {},
                messages,
                lambda m: m.raw_message,
                _message_len(sys_msg) - 1,
            )

    @pytest.mark.asyncio
    async def test_raises_system_and_last_user_error(self):
        """System + last user message exceeds the budget."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 4),
            _mm("user", "u" * 7),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await truncate_prompt(
                tokenizer,
                {},
                messages,
                lambda m: m.raw_message,
                _message_len(messages[0]),
            )

    @pytest.mark.asyncio
    async def test_structured_user_content_still_raises_last_user_error(self):
        """Structured content does not change SystemAndLastUser behavior."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 4),
            _mi(text="describe this"),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await truncate_prompt(
                tokenizer,
                {},
                messages,
                lambda m: m.raw_message,
                _message_len(messages[0]),
            )

    @pytest.mark.asyncio
    async def test_structured_user_content_kept_when_fits(self):
        """Structured user content is kept intact when truncation keeps that message."""
        tokenizer = _make_length_based_tokenizer()

        structured_message = _mi(text="describe this")

        messages = [
            _mm("system", "s" * 3),
            _mm("user", "o" * 8),
            structured_message,
        ]
        max_prompt_tokens = _messages_char_count([messages[0], messages[2]])

        truncated, discarded, used = await truncate_prompt(
            tokenizer, {}, messages, lambda m: m.raw_message, max_prompt_tokens
        )

        assert discarded == [1]
        assert used == _messages_char_count(truncated)
        assert truncated == [messages[0], messages[2]]

    @pytest.mark.asyncio
    async def test_tokenize_called_with_full_list_each_iteration(self):
        """Each truncation step re-sends the full remaining message list."""
        call_payloads: list[list[dict]] = []
        tokenizer = _make_length_based_tokenizer(call_payloads=call_payloads)

        structured_message = _mi(text="describe this")

        messages = [
            _mm("system", "s" * 3),
            _mm("user", "u" * 8),
            structured_message,
        ]
        max_prompt_tokens = _messages_char_count([messages[0], messages[2]])

        await truncate_prompt(
            tokenizer, {}, messages, lambda m: m.raw_message, max_prompt_tokens
        )

        assert len(call_payloads[0]) == 3
        assert len(call_payloads[1]) == 2
        assert call_payloads[1][0] == messages[0].raw_message
        assert call_payloads[1][1] == messages[2].raw_message


class TestVllmToolCallCascade:
    @pytest.mark.asyncio
    async def test_assistant_tool_calls_cascade_removes_tool_messages(self):
        """Dropping assistant(tool_calls) also drops following tool chain."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 3),
            _ma(content=None, function_name="f" * 2, arguments="a" * 6),
            _mm("tool", "r" * 8, tool_call_id="call_1"),
            _mm("tool", "q" * 8, tool_call_id="call_1"),
            _mm("assistant", "n" * 7),
            _mm("user", "f" * 4),
        ]

        truncated, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count([messages[0], messages[5]]),
        )

        assert sorted(discarded) == [1, 2, 3, 4]
        assert used == _messages_char_count(truncated)
        assert truncated == [messages[0], messages[5]]

    @pytest.mark.asyncio
    async def test_non_tool_call_assistant_no_cascade(self):
        """A plain assistant message (no tool_calls) does not cascade."""
        tokenizer = _make_length_based_tokenizer()

        messages = [
            _mm("system", "s" * 3),
            _mm("assistant", "a" * 9),
            _mm("tool", "t" * 4, tool_call_id="call_1"),
            _mm("user", "u" * 4),
        ]

        _, discarded, used = await truncate_prompt(
            tokenizer,
            {},
            messages,
            lambda m: m.raw_message,
            _messages_char_count([messages[0], messages[2], messages[3]]),
        )

        assert sorted(discarded) == [1]
        assert used == _messages_char_count(
            [messages[0], messages[2], messages[3]]
        )


class TestVllmExtraHeaders:
    @pytest.mark.asyncio
    async def test_extra_headers_included_in_tokenize_request(self):
        """Extra headers (from VLLM_HEADERS_TO_PROXY) are sent with tokenize calls."""
        tokenizer = VllmTokenizer(
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
