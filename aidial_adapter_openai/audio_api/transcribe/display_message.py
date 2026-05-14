import json
import re

_AUDIO_FILE_SIZE_LIMIT_EXCEEDED = "Audio file size exceeds the allowed limit."
_PROVIDER_AUDIO_SIZE_LIMIT_PATTERN = re.compile(
    r"Maximum content size limit \((\d+)\) exceeded \((\d+) bytes read\)"
)


def _format_size_mb(size_bytes: int) -> str:
    return f"{(size_bytes / (1024 * 1024)):.1f}".rstrip("0").rstrip(".")


def invalid_file_format_msg(body: str) -> str:
    try:
        payload, _ = json.JSONDecoder().raw_decode(body.strip())
    except json.JSONDecodeError:
        return body

    if not isinstance(payload, dict):
        return body

    message = payload.get("error", {}).get("message")
    return str(message) if message else body


def file_too_large_msg(body: str) -> str:
    if match := _PROVIDER_AUDIO_SIZE_LIMIT_PATTERN.search(body):
        limit_bytes = int(match.group(1))
        actual_bytes = int(match.group(2))
        return (
            f"Audio file size ({_format_size_mb(actual_bytes)}MB) exceeds "
            f"the {_format_size_mb(limit_bytes)}MB limit."
        )

    return _AUDIO_FILE_SIZE_LIMIT_EXCEEDED
