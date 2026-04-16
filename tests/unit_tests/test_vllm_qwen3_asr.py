from collections.abc import AsyncIterator
from typing import Any

from aidial_adapter_openai.chat_completions.vllm import (
    extract_qwen3_asr_language,
)
from tests.utils.stream import single_choice_chunk


def _make_response(content: str) -> dict:
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "qwen-asr",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _collect_stages(results: list[dict]) -> list[dict]:
    stages: list[dict] = []
    for chunk in results:
        for choice in chunk.get("choices", []):
            cc = choice.get("delta", {}).get("custom_content", {})
            stages.extend(cc.get("stages", []))
    return stages


def _content_per_chunk(results: list[dict]) -> list[str | None]:
    out: list[str | None] = []
    for chunk in results:
        choices = chunk.get("choices", [])
        delta = choices[0].get("delta", {}) if choices else {}
        out.append(delta.get("content"))
    return out


def test_non_streaming_happy_path() -> None:
    result = extract_qwen3_asr_language(
        _make_response("language English<asr_text>hello world")
    )

    message = result["choices"][0]["message"]
    assert message["content"] == "hello world"

    stages = message.get("custom_content", {}).get("stages", [])
    assert len(stages) == 1
    assert stages[0]["name"] == "Language: English"


async def test_streaming_prefix_in_single_delta() -> None:
    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield single_choice_chunk(
            delta={"content": "language French<asr_text>"}
        )
        yield single_choice_chunk(delta={"content": "Bonjour "})
        yield single_choice_chunk(delta={"content": "le "})
        yield single_choice_chunk(delta={"content": "monde"})
        yield single_choice_chunk(delta={}, finish_reason="stop")

    results = [c async for c in extract_qwen3_asr_language(mock_stream())]

    stages = _collect_stages(results)
    assert len(stages) == 1
    assert stages[0]["name"] == "Language: French"

    assert _content_per_chunk(results) == [
        "",
        "Bonjour ",
        "le ",
        "monde",
        None,
    ]


async def test_streaming_prefix_split_across_two_deltas() -> None:
    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield single_choice_chunk(delta={"content": "language Eng"})
        yield single_choice_chunk(
            delta={"content": "lish<asr_text>recognized speech"}
        )
        yield single_choice_chunk(delta={"content": " continues"})
        yield single_choice_chunk(delta={}, finish_reason="stop")

    results = [c async for c in extract_qwen3_asr_language(mock_stream())]

    stages = _collect_stages(results)
    assert len(stages) == 1
    assert stages[0]["name"] == "Language: English"

    assert _content_per_chunk(results) == [
        None,
        "recognized speech",
        " continues",
        None,
    ]


async def test_streaming_language_starts_then_diverges() -> None:
    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield single_choice_chunk(delta={"content": "lang"})
        yield single_choice_chunk(
            delta={"content": "uage is important for communication"}
        )
        yield single_choice_chunk(delta={"content": " and "})
        yield single_choice_chunk(delta={"content": "understanding"})
        yield single_choice_chunk(delta={}, finish_reason="stop")

    results = [c async for c in extract_qwen3_asr_language(mock_stream())]

    assert _collect_stages(results) == []

    assert _content_per_chunk(results) == [
        None,
        "language is important for communication",
        " and ",
        "understanding",
        None,
    ]


async def test_streaming_no_prefix_from_start() -> None:
    async def mock_stream() -> AsyncIterator[dict[str, Any]]:
        yield single_choice_chunk(delta={"content": "Hello "})
        yield single_choice_chunk(delta={"content": "beautiful "})
        yield single_choice_chunk(delta={"content": "world"})
        yield single_choice_chunk(delta={}, finish_reason="stop")

    results = [c async for c in extract_qwen3_asr_language(mock_stream())]

    assert _collect_stages(results) == []

    assert _content_per_chunk(results) == [
        "Hello ",
        "beautiful ",
        "world",
        None,
    ]
