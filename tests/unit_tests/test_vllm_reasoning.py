"""Unit tests for vLLM reasoning content transformation."""

from typing import Any, AsyncIterator

from aidial_adapter_openai.chat_completions.vllm import extract_reasoning
from tests.utils.stream import single_choice_chunk


def _make_response(
    reasoning: str | None = None,
    content: str | None = None,
    finish_reason: str = "stop",
) -> dict:
    """Create a vLLM non-streaming response."""
    message: dict = {"role": "assistant"}
    if reasoning is not None:
        message["reasoning"] = reasoning
    if content is not None:
        message["content"] = content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "deepseek-r1",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }


class TestNonStreamingVllmReasoning:
    """Test non-streaming vLLM reasoning extraction."""

    def test_reasoning_only(self) -> None:
        """Reasoning without content should create a stage."""
        result = extract_reasoning(
            _make_response(
                reasoning="Let me think about this...",
                content=None,
            )
        )
        message = result["choices"][0]["message"]
        assert "reasoning" not in message
        assert message.get("content") is None
        stage = message["custom_content"]["stages"][0]
        assert stage["content"] == "Let me think about this..."
        assert stage["name"] == "Reasoning"
        assert "status" not in stage  # Closing comes in next chunk

    def test_reasoning_and_content(self) -> None:
        """Both reasoning and content should extract reasoning into stage."""
        result = extract_reasoning(
            _make_response(
                reasoning="9.11 is greater because 9.11 > 9.8",
                content="9.11 is greater than 9.8",
            )
        )
        message = result["choices"][0]["message"]
        assert "reasoning" not in message
        assert message["content"] == "9.11 is greater than 9.8"
        stage = message["custom_content"]["stages"][0]
        assert stage["content"] == "9.11 is greater because 9.11 > 9.8"
        assert stage["name"] == "Reasoning"
        assert "status" not in stage  # Closing comes in next chunk

    def test_content_only(self) -> None:
        """No reasoning should not create stage."""
        result = extract_reasoning(
            _make_response(reasoning=None, content="Just an answer")
        )
        message = result["choices"][0]["message"]
        assert message["content"] == "Just an answer"
        assert "custom_content" not in message

    def test_empty_response(self) -> None:
        """Empty response should pass through unchanged."""
        result = extract_reasoning(_make_response(reasoning=None, content=None))
        message = result["choices"][0]["message"]
        assert "custom_content" not in message

    def test_multiple_choices_independent_tracking(self) -> None:
        """Each choice should independently track whether it has reasoning."""
        response = {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": "deepseek-r1",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "reasoning": "Reasoning for choice 0",
                        "content": "Answer 0",
                    },
                    "finish_reason": "stop",
                },
                {
                    "index": 1,
                    "message": {
                        "role": "assistant",
                        "content": "Answer 1",
                    },
                    "finish_reason": "stop",
                },
            ],
        }
        result = extract_reasoning(response)

        # Choice 0 has reasoning
        msg0 = result["choices"][0]["message"]
        assert "reasoning" not in msg0
        assert msg0["content"] == "Answer 0"
        assert (
            msg0["custom_content"]["stages"][0]["content"]
            == "Reasoning for choice 0"
        )

        # Choice 1 has no reasoning
        msg1 = result["choices"][1]["message"]
        assert msg1["content"] == "Answer 1"
        assert "custom_content" not in msg1


class TestStreamingVllmReasoning:
    """Test streaming vLLM reasoning extraction."""

    async def test_reasoning_then_content(self) -> None:
        """Streaming: reasoning chunks followed by content chunks."""

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            # First chunk: opens reasoning stage
            yield single_choice_chunk(
                delta={
                    "role": "assistant",
                    "reasoning": "Let me think",
                }
            )
            # Second chunk: continues reasoning
            yield single_choice_chunk(delta={"reasoning": " about this"})
            # Third chunk: content (reasoning ends)
            yield single_choice_chunk(delta={"content": "The answer"})
            # Final chunk: close
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [chunk async for chunk in extract_reasoning(mock_stream())]

        # First chunk: opens reasoning stage
        delta0 = results[0]["choices"][0]["delta"]
        assert delta0.get("role") == "assistant"
        assert "reasoning" not in delta0
        stage0 = delta0["custom_content"]["stages"][0]
        assert stage0["name"] == "Reasoning"
        assert stage0["content"] == "Let me think"
        assert stage0["index"] == 0
        assert "status" not in stage0

        # Second chunk: continues reasoning
        delta1 = results[1]["choices"][0]["delta"]
        assert "reasoning" not in delta1
        stage1 = delta1["custom_content"]["stages"][0]
        assert "name" not in stage1
        assert stage1["content"] == " about this"
        assert stage1["index"] == 0
        assert "status" not in stage1

        # Third chunk: content (no stage, reasoning field is removed)
        delta2 = results[2]["choices"][0]["delta"]
        assert delta2["content"] == "The answer"
        assert "custom_content" not in delta2

        # Final chunk: closes stage
        delta3 = results[3]["choices"][0]["delta"]
        # finish_reason is in the choice, not delta
        assert results[3]["choices"][0]["finish_reason"] == "stop"
        stage3 = delta3["custom_content"]["stages"][0]
        assert stage3["status"] == "completed"
        assert stage3["index"] == 0
        assert "content" not in stage3
        assert "name" not in stage3

    async def test_reasoning_and_close_same_chunk(self) -> None:
        """Streaming: reasoning and finish_reason in same chunk opens stage but doesn't close yet."""

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(
                delta={"reasoning": "Quick thought"},
                finish_reason="stop",
            )
            # Need a second chunk to close (without reasoning but with finish_reason still set)
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [chunk async for chunk in extract_reasoning(mock_stream())]

        # First chunk: opens the stage
        stage0 = results[0]["choices"][0]["delta"]["custom_content"]["stages"][
            0
        ]
        assert stage0["name"] == "Reasoning"
        assert stage0["content"] == "Quick thought"
        assert stage0["index"] == 0
        assert "status" not in stage0  # Not closed yet

        # Second chunk: closes the stage
        stage1 = results[1]["choices"][0]["delta"]["custom_content"]["stages"][
            0
        ]
        assert stage1["status"] == "completed"
        assert stage1["index"] == 0
        assert "content" not in stage1
        assert "name" not in stage1

    async def test_content_only_streaming(self) -> None:
        """Streaming: content without reasoning."""

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(delta={"content": "Just text"})
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [chunk async for chunk in extract_reasoning(mock_stream())]

        assert results[0]["choices"][0]["delta"]["content"] == "Just text"
        assert "custom_content" not in results[0]["choices"][0]["delta"]
        assert "custom_content" not in results[1]["choices"][0]["delta"]

    async def test_multiple_choices_streaming(self) -> None:
        """Streaming: multiple choices with independent reasoning tracking."""

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            # Choice 0: has reasoning
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "deepseek-r1",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"reasoning": "Thought 0"},
                        "finish_reason": None,
                    },
                ],
            }
            # Choice 1: has content
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "deepseek-r1",
                "choices": [
                    {
                        "index": 1,
                        "delta": {"content": "Answer 1"},
                        "finish_reason": None,
                    },
                ],
            }
            # Both close
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "deepseek-r1",
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ],
            }

        results = [chunk async for chunk in extract_reasoning(mock_stream())]

        # Choice 0 should have reasoning stage opened
        delta0_0 = results[0]["choices"][0]["delta"]
        assert "reasoning" not in delta0_0
        assert delta0_0["custom_content"]["stages"][0]["name"] == "Reasoning"

        # Choice 1 should have content
        delta1_1 = results[1]["choices"][0]["delta"]
        assert delta1_1["content"] == "Answer 1"
        assert "custom_content" not in delta1_1

        # Choice 0 should close properly
        delta2_0 = results[2]["choices"][0]["delta"]
        assert delta2_0["custom_content"]["stages"][0]["status"] == "completed"

        # Choice 1 never had reasoning, so no stage is created on close
        delta2_1 = results[2]["choices"][1]["delta"]
        assert "custom_content" not in delta2_1
