"""Unit tests for GPT-4o audio completions support."""

from typing import Any, AsyncIterator, Callable

import pytest

from aidial_adapter_openai.chat_completions.gpt_audio import (
    extract_audio_content,
)
from tests.utils.stream import single_choice_chunk


def _make_response(audio_obj: dict[str, str] | None = None) -> dict[str, Any]:
    """Helper to create a non-streaming response."""
    message: dict = {"role": "assistant"}
    if audio_obj:
        message["audio"] = audio_obj
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 1234567890,
        "model": "gpt-4o-audio-preview",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }


class TestNonStreamingAudio:
    """Test suite for non-streaming audio responses."""

    @pytest.mark.parametrize(
        "audio_obj,has_attachment,has_stage,attachment_check,stage_check",
        [
            # Both data and transcript
            (
                {"data": "audio_data", "transcript": "Audio transcript"},
                True,
                True,
                lambda a: a["data"] == "audio_data",
                lambda s: s["content"] == "Audio transcript"
                and s["name"] == "Audio transcript"
                and s["status"] == "completed",
            ),
            # Data only
            (
                {"data": "audio_data_only"},
                True,
                False,
                lambda a: a["data"] == "audio_data_only",
                None,
            ),
            # Transcript only
            (
                {"transcript": "Just transcript"},
                False,
                True,
                None,
                lambda s: s["content"] == "Just transcript"
                and s["name"] == "Audio transcript",
            ),
            # No audio field
            (None, False, False, None, None),
        ],
    )
    def test_audio_extraction_scenarios(
        self,
        audio_obj: dict[str, str] | None,
        has_attachment: bool,
        has_stage: bool,
        attachment_check: Callable[[dict[str, Any]], bool],
        stage_check: Callable[[dict[str, Any]], bool],
    ) -> None:
        """Test various combinations of audio data and transcript."""
        request_body = {"audio": {"format": "mp3"}}
        response = _make_response(audio_obj)
        result = extract_audio_content(response, request_body)

        message = result["choices"][0]["message"]
        if audio_obj:
            assert message["audio"] == audio_obj
        else:
            assert "audio" not in message

        if has_attachment or has_stage:
            cc = message["custom_content"]
            assert ("attachments" in cc) == has_attachment
            assert ("stages" in cc) == has_stage

            if has_attachment:
                assert cc["attachments"][0]["type"] == "audio/mpeg"
                assert attachment_check(cc["attachments"][0])

            if has_stage:
                assert stage_check(cc["stages"][0])
        else:
            assert "custom_content" not in message

    @pytest.mark.parametrize(
        "audio_format,expected_type",
        [
            ("mp3", "audio/mpeg"),
            ("wav", "audio/wav"),
            ("flac", "audio/flac"),
            ("opus", "audio/opus"),
            ("pcm16", "audio/L16"),
            ("unknown", "audio/unknown"),
        ],
    )
    def test_audio_formats(self, audio_format: str, expected_type: str) -> None:
        """Test different audio format content type mappings."""
        request_body = {"audio": {"format": audio_format}}
        response = _make_response({"data": "test"})
        result = extract_audio_content(response, request_body)

        attachment = result["choices"][0]["message"]["custom_content"][
            "attachments"
        ][0]
        assert attachment["type"] == expected_type

    def test_default_audio_format(self) -> None:
        """Test mp3 is used as default when format not specified."""
        response = _make_response({"data": "test"})
        result = extract_audio_content(response, {})

        attachment = result["choices"][0]["message"]["custom_content"][
            "attachments"
        ][0]
        assert attachment["type"] == "audio/mpeg"

    def test_multiple_choices(self) -> None:
        """Test handling of multiple choices."""
        request_body = {"audio": {"format": "mp3"}}
        response = {
            "id": "test",
            "object": "chat.completion",
            "created": 123,
            "model": "gpt-4o",
            "choices": [
                {
                    "index": i,
                    "message": {
                        "role": "assistant",
                        "audio": {
                            "data": f"data_{i}",
                            "transcript": f"transcript_{i}",
                        },
                    },
                    "finish_reason": "stop",
                }
                for i in range(2)
            ],
        }

        result = extract_audio_content(response, request_body)

        for i, choice in enumerate(result["choices"]):
            cc = choice["message"]["custom_content"]
            assert cc["attachments"][0]["data"] == f"data_{i}"
            assert cc["stages"][0]["content"] == f"transcript_{i}"


class TestStreamingAudio:
    """Test suite for streaming audio responses."""

    async def test_complete_streaming_flow(self) -> None:
        """Test full streaming with stage opening, continuation, and closing."""
        request_body = {"audio": {"format": "mp3"}}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(delta={"role": "assistant"})
            yield single_choice_chunk(
                delta={"audio": {"data": "d1", "transcript": "First"}}
            )
            yield single_choice_chunk(
                delta={"audio": {"data": "d2", "transcript": " second"}}
            )
            yield single_choice_chunk(delta={}, finish_reason="stop")

        results = [
            chunk
            async for chunk in extract_audio_content(
                mock_stream(), request_body
            )
        ]

        # First audio chunk: opens stage and attachment
        cc1 = results[1]["choices"][0]["delta"]["custom_content"]
        assert cc1["stages"][0]["name"] == "Audio transcript"
        assert cc1["stages"][0]["index"] == 0
        assert cc1["attachments"][0]["data"] == "d1"
        assert cc1["attachments"][0]["title"] == "Audio"
        assert cc1["attachments"][0]["type"] == "audio/mpeg"
        assert cc1["attachments"][0]["index"] == 0

        # Second audio chunk: continues (no name/status for stage, no title/type for attachment)
        cc2 = results[2]["choices"][0]["delta"]["custom_content"]
        assert "name" not in cc2["stages"][0]
        assert "status" not in cc2["stages"][0]
        assert cc2["attachments"][0]["data"] == "d2"
        assert "title" not in cc2["attachments"][0]
        assert "type" not in cc2["attachments"][0]
        assert cc2["attachments"][0]["index"] == 0

        # Final chunk: no custom_content (no audio)
        assert "custom_content" not in results[3]["choices"][0]["delta"]

    async def test_streaming_with_finish(self) -> None:
        """Test audio in final chunk (opening and closing together)."""
        request_body = {"audio": {"format": "mp3"}}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield single_choice_chunk(delta={"role": "assistant"})
            yield single_choice_chunk(
                delta={"audio": {"transcript": "Done"}}, finish_reason="stop"
            )

        results = [
            chunk
            async for chunk in extract_audio_content(
                mock_stream(), request_body
            )
        ]

        stage = results[1]["choices"][0]["delta"]["custom_content"]["stages"][0]
        assert stage["name"] == "Audio transcript"
        assert stage["status"] == "completed"

    @pytest.mark.parametrize(
        "audio_obj,has_attachment,has_stage",
        [
            ({"data": "d", "transcript": "t"}, True, True),
            ({"data": "d"}, True, False),
            ({"transcript": "t"}, False, True),
            (None, False, False),
        ],
    )
    async def test_streaming_audio_combinations(
        self,
        audio_obj: dict[str, str] | None,
        has_attachment: bool,
        has_stage: bool,
    ) -> None:
        """Test streaming with different audio content combinations."""
        request_body = {"audio": {"format": "mp3"}}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            delta = {"audio": audio_obj} if audio_obj else {}
            yield single_choice_chunk(delta=delta)

        results = [
            chunk
            async for chunk in extract_audio_content(
                mock_stream(), request_body
            )
        ]

        delta = results[0]["choices"][0]["delta"]
        if has_attachment or has_stage:
            cc = delta["custom_content"]
            assert ("attachments" in cc) == has_attachment
            assert ("stages" in cc) == has_stage
        else:
            assert "custom_content" not in delta

    async def test_streaming_multiple_choices(self) -> None:
        """Test independent stage tracking for multiple choices."""
        request_body = {"audio": {"format": "mp3"}}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            # Choice 0 opens, then continues
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"audio": {"transcript": "C0"}},
                        "finish_reason": None,
                    }
                ],
            }
            # Choice 1 opens
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 1,
                        "delta": {"audio": {"transcript": "C1"}},
                        "finish_reason": None,
                    }
                ],
            }
            # Choice 0 continues (no opening)
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 123,
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"audio": {"transcript": " more"}},
                        "finish_reason": None,
                    }
                ],
            }

        results = [
            chunk
            async for chunk in extract_audio_content(
                mock_stream(), request_body
            )
        ]

        # Choice 0 first chunk: has name (opening)
        assert (
            "name"
            in results[0]["choices"][0]["delta"]["custom_content"]["stages"][0]
        )
        # Choice 1 first chunk: has name (opening)
        assert (
            "name"
            in results[1]["choices"][0]["delta"]["custom_content"]["stages"][0]
        )
        # Choice 0 second chunk: no name (continuing)
        assert (
            "name"
            not in results[2]["choices"][0]["delta"]["custom_content"][
                "stages"
            ][0]
        )

    async def test_empty_choices_list(self) -> None:
        """Test graceful handling of empty choices list."""
        request_body = {"audio": {"format": "mp3"}}

        async def mock_stream() -> AsyncIterator[dict[str, Any]]:
            yield {
                "id": "test",
                "object": "chat.completion.chunk",
                "choices": [],
            }
            yield single_choice_chunk(delta={"content": "Test"})

        results = [
            chunk
            async for chunk in extract_audio_content(
                mock_stream(), request_body
            )
        ]

        assert len(results) == 2
        assert results[0]["choices"] == []
