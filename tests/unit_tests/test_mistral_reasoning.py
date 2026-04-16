"""Unit tests for Magistral reasoning content transformation."""

from collections.abc import AsyncIterator
from typing import Any

from aidial_adapter_openai.chat_completions.mistral import (
    extract_reasoning_content,
)
from tests.utils.stream import single_choice_chunk


def _make_response(content: Any = None, finish_reason: str = "stop") -> dict:
    message: dict = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "magistral-medium-latest",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason}
        ],
    }


class TestNonStreamingMistralReasoning:
    def test_text_only_content(self) -> None:
        result = extract_reasoning_content(
            _make_response(content=[{"type": "text", "text": "Hello world"}])
        )
        message = result["choices"][0]["message"]
        assert message["content"] == "Hello world"
        assert "custom_content" not in message

    def test_thinking_only_content(self) -> None:
        result = extract_reasoning_content(
            _make_response(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [
                            {"type": "text", "text": "Let me think..."}
                        ],
                    }
                ]
            )
        )
        message = result["choices"][0]["message"]
        assert message["content"] is None
        stage = message["custom_content"]["stages"][0]
        assert stage["content"] == "Let me think..."
        assert stage["name"] == "Reasoning"
        assert stage["status"] == "completed"

    def test_thinking_and_text_content(self) -> None:
        result = extract_reasoning_content(
            _make_response(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [
                            {"type": "text", "text": "Reasoning here"}
                        ],
                    },
                    {"type": "text", "text": "Answer here"},
                ]
            )
        )
        message = result["choices"][0]["message"]
        assert message["content"] == "Answer here"
        stage = message["custom_content"]["stages"][0]
        assert stage["content"] == "Reasoning here"
        assert stage["name"] == "Reasoning"
        assert stage["status"] == "completed"

    def test_thinking_as_multiple_items_in_list(self) -> None:
        result = extract_reasoning_content(
            _make_response(
                content=[
                    {
                        "type": "thinking",
                        "thinking": [
                            {"type": "text", "text": "Part A"},
                            {"type": "text", "text": " Part B"},
                        ],
                    },
                    {"type": "text", "text": "Answer"},
                ]
            )
        )
        message = result["choices"][0]["message"]
        assert message["content"] == "Answer"
        stage = message["custom_content"]["stages"][0]
        assert stage["content"] == "Part A Part B"
        assert stage["name"] == "Reasoning"
        assert stage["status"] == "completed"

    def test_multiple_text_parts_concatenated(self) -> None:
        result = extract_reasoning_content(
            _make_response(
                content=[
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": " world"},
                ]
            )
        )
        assert result["choices"][0]["message"]["content"] == "Hello world"

    def test_empty_list_content(self) -> None:
        result = extract_reasoning_content(_make_response(content=[]))
        message = result["choices"][0]["message"]
        assert message["content"] is None
        assert "custom_content" not in message

    def test_string_content_passthrough(self) -> None:
        result = extract_reasoning_content(_make_response(content="Simple"))
        message = result["choices"][0]["message"]
        assert message["content"] == "Simple"
        assert "custom_content" not in message

    def test_none_content_passthrough(self) -> None:
        result = extract_reasoning_content(_make_response())
        message = result["choices"][0]["message"]
        assert "content" not in message
        assert "custom_content" not in message


class TestStreamingMistralReasoning:
    async def test_thinking_then_text_then_close(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(
                delta={
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": "Part 1"}],
                        }
                    ],
                }
            )
            yield single_choice_chunk(
                delta={
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [{"type": "text", "text": " Part 2"}],
                        }
                    ]
                }
            )
            yield single_choice_chunk(
                delta={"content": [{"type": "text", "text": "Answer"}]}
            )
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [
            chunk async for chunk in extract_reasoning_content(mock_stream())
        ]

        # First thinking chunk: opens stage
        delta0 = results[0]["choices"][0]["delta"]
        assert delta0["content"] is None
        stage0 = delta0["custom_content"]["stages"][0]
        assert stage0["name"] == "Reasoning"
        assert stage0["content"] == "Part 1"
        assert stage0["index"] == 0
        assert "status" not in stage0

        # Second thinking chunk: continues stage (no name)
        delta1 = results[1]["choices"][0]["delta"]
        assert delta1["content"] is None
        stage1 = delta1["custom_content"]["stages"][0]
        assert "name" not in stage1
        assert stage1["content"] == " Part 2"
        assert stage1["index"] == 0
        assert "status" not in stage1

        # Text chunk: content is set, no stage
        delta2 = results[2]["choices"][0]["delta"]
        assert delta2["content"] == "Answer"
        assert "custom_content" not in delta2

        # Final chunk: closes stage
        delta3 = results[3]["choices"][0]["delta"]
        stage3 = delta3["custom_content"]["stages"][0]
        assert stage3["status"] == "completed"
        assert stage3["index"] == 0
        assert "content" not in stage3
        assert "name" not in stage3

    async def test_thinking_and_close_in_same_chunk(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(
                delta={
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": [
                                {"type": "text", "text": "Only thought"}
                            ],
                        }
                    ]
                },
                finish_reason="stop",
            )

        results = [
            chunk async for chunk in extract_reasoning_content(mock_stream())
        ]

        stage = results[0]["choices"][0]["delta"]["custom_content"]["stages"][0]
        assert stage["name"] == "Reasoning"
        assert stage["content"] == "Only thought"
        assert stage["status"] == "completed"
        assert stage["index"] == 0

    async def test_string_content_passthrough_streaming(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(delta={"content": "Hello"})

        results = [
            chunk async for chunk in extract_reasoning_content(mock_stream())
        ]
        delta = results[0]["choices"][0]["delta"]
        assert delta["content"] == "Hello"
        assert "custom_content" not in delta

    async def test_no_thinking_no_stage(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(
                delta={"content": [{"type": "text", "text": "Hi"}]}
            )
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [
            chunk async for chunk in extract_reasoning_content(mock_stream())
        ]
        assert results[0]["choices"][0]["delta"]["content"] == "Hi"
        assert "custom_content" not in results[0]["choices"][0]["delta"]
        assert "custom_content" not in results[1]["choices"][0]["delta"]

    async def test_multiple_choices_independent_tracking(self) -> None:
        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            # Choice 0 opens stage
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "magistral-medium-latest",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": [
                                        {"type": "text", "text": "C0 thought"}
                                    ],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            # Choice 1 opens stage
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "magistral-medium-latest",
                "choices": [
                    {
                        "index": 1,
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": [
                                        {"type": "text", "text": "C1 thought"}
                                    ],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }
            # Choice 0 continues stage (no name expected)
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "magistral-medium-latest",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "content": [
                                {
                                    "type": "thinking",
                                    "thinking": [
                                        {"type": "text", "text": " more"}
                                    ],
                                }
                            ]
                        },
                        "finish_reason": None,
                    }
                ],
            }

        results = [
            chunk async for chunk in extract_reasoning_content(mock_stream())
        ]

        # Choice 0 first appearance: has name
        assert (
            "name"
            in results[0]["choices"][0]["delta"]["custom_content"]["stages"][0]
        )
        # Choice 1 first appearance: has name
        assert (
            "name"
            in results[1]["choices"][0]["delta"]["custom_content"]["stages"][0]
        )
        # Choice 0 second appearance: no name (continuing)
        assert (
            "name"
            not in results[2]["choices"][0]["delta"]["custom_content"][
                "stages"
            ][0]
        )
