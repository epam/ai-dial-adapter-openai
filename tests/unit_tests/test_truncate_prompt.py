import pytest
from aidial_sdk.exceptions import (
    TruncatePromptSystemAndLastUserError,
    TruncatePromptSystemError,
)

from aidial_adapter_openai.utils.truncate_prompt import truncate_prompt


def _message(role: str, tokens: int) -> dict:
    return {"role": role, "tokens": tokens}


class LengthTokenizerStub:
    def __init__(self, *, call_payloads: list[list[dict]] | None = None):
        self.call_payloads = call_payloads

    async def tokenize(self, request: dict) -> int:
        raw_messages = request.get("messages", [])
        if self.call_payloads is not None:
            self.call_payloads.append(list(raw_messages))

        per_request_tokens = request.get("per_request_tokens", 0)
        message_tokens = sum(m.get("tokens", 0) for m in raw_messages)
        return per_request_tokens + message_tokens


async def _truncate_prompt(
    per_request_tokens: int, messages: list, max_prompt_tokens: int
):
    return await truncate_prompt(
        LengthTokenizerStub(),
        {"per_request_tokens": per_request_tokens},
        messages,
        lambda m: m,
        max_prompt_tokens,
    )


class TestTruncatePrompt:
    @pytest.mark.asyncio
    async def test_fits_without_truncation(self):
        messages = [_message("system", 5), _message("user", 4)]

        truncated, discarded, used = await _truncate_prompt(0, messages, 9)

        assert discarded == []
        assert used == 9
        assert truncated == messages

    @pytest.mark.asyncio
    async def test_drops_oldest_non_system_message(self):
        messages = [
            _message("system", 3),
            _message("user", 10),
            _message("assistant", 5),
            _message("user", 8),
        ]

        truncated, discarded, used = await _truncate_prompt(0, messages, 16)

        assert discarded == [1]
        assert used == 16
        assert truncated == [messages[0], messages[2], messages[3]]

    @pytest.mark.asyncio
    async def test_drops_multiple_messages(self):
        messages = [
            _message("system", 3),
            _message("user", 9),
            _message("assistant", 8),
            _message("user", 7),
        ]

        truncated, discarded, used = await _truncate_prompt(0, messages, 10)

        assert sorted(discarded) == [1, 2]
        assert used == 10
        assert truncated == [messages[0], messages[3]]

    @pytest.mark.asyncio
    async def test_no_system_message_truncation_succeeds(self):
        messages = [
            _message("user", 9),
            _message("assistant", 8),
            _message("user", 6),
            _message("assistant", 5),
            _message("user", 3),
        ]

        truncated, discarded, used = await _truncate_prompt(0, messages, 14)

        assert sorted(discarded) == [0, 1]
        assert used == 14
        assert truncated == [messages[2], messages[3], messages[4]]

    @pytest.mark.asyncio
    async def test_no_system_message_last_message_too_big(self):
        messages = [
            _message("user", 3),
            _message("assistant", 3),
            _message("user", 11),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await _truncate_prompt(0, messages, 3)

    @pytest.mark.asyncio
    async def test_raises_system_error(self):
        messages = [_message("system", 11)]

        with pytest.raises(TruncatePromptSystemError):
            await _truncate_prompt(0, messages, 10)

    @pytest.mark.asyncio
    async def test_raises_system_and_last_user_error(self):
        messages = [
            _message("system", 4),
            _message("user", 7),
        ]

        with pytest.raises(TruncatePromptSystemAndLastUserError):
            await _truncate_prompt(0, messages, 4)

    @pytest.mark.asyncio
    async def test_tokenize_called_with_full_list_each_iteration(self):
        call_payloads: list[list[dict]] = []
        tokenizer = LengthTokenizerStub(call_payloads=call_payloads)

        # system can appear mid-conversation, not just at the start
        messages = [
            _message("user", 8),
            _message("system", 3),
            _message("user", 7),
        ]

        await truncate_prompt(
            tokenizer,
            {"per_request_tokens": 2},
            messages,
            lambda m: m,
            12,
        )

        assert len(call_payloads[0]) == 3
        assert len(call_payloads[1]) == 2
        assert call_payloads[1][0] == messages[1]  # system kept
        assert call_payloads[1][1] == messages[2]  # last user kept


class TestTruncatePromptToolCallCascade:
    @pytest.mark.asyncio
    async def test_assistant_tool_calls_cascade_removes_tool_messages(self):
        messages = [
            _message("system", 3),
            _message("user", 5),
            {**_message("assistant", 8), "tool_calls": [{}]},
            _message("tool", 8),
            _message("tool", 8),
            _message("assistant", 7),
            _message("user", 4),
        ]

        truncated, discarded, used = await _truncate_prompt(0, messages, 7)

        assert sorted(discarded) == [1, 2, 3, 4, 5]
        assert used == 7
        assert truncated == [messages[0], messages[6]]

    @pytest.mark.asyncio
    async def test_non_tool_call_assistant_no_cascade(self):
        messages = [
            _message("system", 3),
            _message("user", 9),
            _message("assistant", 6),  # no tool_calls — no cascade on drop
            _message("tool", 4),
            _message("user", 4),
        ]

        _, discarded, used = await _truncate_prompt(0, messages, 11)

        # assistant (6) dropped without cascading tool (4); tool dropped on its own next step
        assert sorted(discarded) == [1, 2]
        assert used == 11
