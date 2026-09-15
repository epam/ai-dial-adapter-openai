"""Unit tests for gpt-oss reasoning content transformation."""

from collections.abc import AsyncIterator
from typing import Any

from aidial_adapter_openai.chat_completions.gpt_oss import (
    extract_reasoning_tokens,
)
from tests.utils.stream import single_choice_chunk


def _make_response(
    reasoning_content: str | None = None,
    content: str | None = None,
    finish_reason: str = "stop",
) -> dict:
    message: dict = {"role": "assistant"}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if content is not None:
        message["content"] = content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-oss-120b",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }


class TestNonStreamingGptOssReasoning:
    def test_reasoning_content_is_reported_in_message_and_stage(self) -> None:
        result = extract_reasoning_tokens(
            _make_response(
                reasoning_content="Let me think...", content="The answer"
            )
        )
        message = result["choices"][0]["message"]
        assert message["reasoning_content"] == "Let me think..."
        assert message["content"] == "The answer"
        assert message["custom_content"]["stages"] == [
            {"content": "Let me think...", "name": "Reasoning"}
        ]

    def test_no_reasoning_content(self) -> None:
        result = extract_reasoning_tokens(
            _make_response(content="Just an answer")
        )
        message = result["choices"][0]["message"]
        assert "reasoning_content" not in message
        assert "custom_content" not in message


class TestStreamingGptOssReasoning:
    async def test_reasoning_content_is_preserved_in_deltas(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(
                delta={"role": "assistant", "reasoning_content": "Let me "}
            )
            yield single_choice_chunk(delta={"reasoning_content": "think"})
            yield single_choice_chunk(delta={"content": "The answer"})
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [
            chunk async for chunk in extract_reasoning_tokens(mock_stream())
        ]
        deltas = [result["choices"][0]["delta"] for result in results]

        assert deltas[0]["reasoning_content"] == "Let me "
        assert deltas[0]["custom_content"]["stages"] == [
            {"content": "Let me ", "index": 0, "name": "Reasoning"}
        ]

        assert deltas[1]["reasoning_content"] == "think"
        assert deltas[1]["custom_content"]["stages"] == [
            {"content": "think", "index": 0}
        ]

        assert "reasoning_content" not in deltas[2]
        assert "custom_content" not in deltas[2]

        assert "reasoning_content" not in deltas[3]
        assert deltas[3]["custom_content"]["stages"] == [
            {"index": 0, "status": "completed"}
        ]
