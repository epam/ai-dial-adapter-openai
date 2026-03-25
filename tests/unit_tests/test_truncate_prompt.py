from typing import Any

import pytest
from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.truncate_prompt import truncate_prompt

_TEST_IMAGE_URL = "i" * 10


def _mm(role: str, content, **extra) -> MultiModalMessage:
    raw_message = {"role": role, "content": content, **extra}
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


class LengthTokenizerStub:
    """Protocol-compatible tokenizer stub for truncate_prompt tests."""

    def __init__(self, *, call_payloads: list[list[dict]] | None = None):
        self.call_payloads = call_payloads

    async def tokenize(self, request: dict[str, Any]) -> int:
        raw_messages = request.get("messages", [])
        if self.call_payloads is not None:
            # Store a shallow copy to protect against accidental mutation.
            self.call_payloads.append(list(raw_messages))

        wrapped = [MultiModalMessage(raw_message=m) for m in raw_messages]
        return _messages_char_count(wrapped)


class TestTruncatePrompt:
    @pytest.mark.asyncio
    async def test_fits_without_truncation(self):
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
        call_payloads: list[list[dict]] = []
        tokenizer = LengthTokenizerStub(call_payloads=call_payloads)

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


class TestTruncatePromptToolCallCascade:
    @pytest.mark.asyncio
    async def test_assistant_tool_calls_cascade_removes_tool_messages(self):
        tokenizer = LengthTokenizerStub()

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
        tokenizer = LengthTokenizerStub()

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
