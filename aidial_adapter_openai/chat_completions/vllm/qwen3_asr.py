import re
from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel, Field

from aidial_adapter_openai.utils.streaming import map_stream

_LANGUAGE_PREFIX_RE = re.compile(
    r"^\s*language\s+([^<\r\n]+?)\s*<asr_text>\s*(.*)$",
    re.IGNORECASE | re.DOTALL,
)

_LANGUAGE_HEADER = "language "

_MAX_PREFIX_LEN = 30


class _Qwen3AsrResponseTransformer(BaseModel):
    """Extract leading Qwen3-ASR language metadata from vLLM content.

    Supported prefix format:
        `language English<asr_text>recognized text`

    References:
        - vLLM guide: https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-ASR.html
        - Model maintainer's API call examples: https://github.com/QwenLM/Qwen3-ASR#vllm-backend
        - Upstream parsing logic: https://github.com/QwenLM/Qwen3-ASR/blob/c17a131fe028b2e428b6e80a33d30bb4fa57b8df/qwen_asr/inference/utils.py#L403
    """

    streaming: bool
    buffers: dict[int, str] = Field(default_factory=dict)
    resolved: set[int] = Field(default_factory=set)

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    def __call__(self, chunk: dict) -> dict:
        for choice in chunk.get("choices") or []:
            choice_index = int(choice.get("index", 0))
            message = choice.get(self.message_key)
            if not isinstance(message, dict):
                continue

            self._transform_choice(
                choice=choice,
                message=message,
                choice_index=choice_index,
            )

        return chunk

    def _transform_choice(
        self,
        *,
        choice: dict,
        message: dict,
        choice_index: int,
    ) -> None:
        if self.streaming:
            self._transform_streaming_choice(
                choice=choice,
                message=message,
                choice_index=choice_index,
            )
            return

        content = message.get("content")
        if not isinstance(content, str):
            return

        parsed = _parse_language_prefix(content)
        if parsed is None:
            return

        language, text = parsed
        message["content"] = text
        _append_language_stage(message, language, streaming=False)

    def _transform_streaming_choice(
        self,
        *,
        choice: dict,
        message: dict,
        choice_index: int,
    ) -> None:
        if choice_index in self.resolved:
            return

        content = message.get("content")
        if isinstance(content, str):
            candidate = self.buffers.get(choice_index, "") + content

            parsed = _parse_language_prefix(candidate)
            if parsed is not None:
                language, text = parsed
                message["content"] = text
                _append_language_stage(message, language, streaming=True)
                self.resolved.add(choice_index)
                self.buffers.pop(choice_index, None)
                return

            if _could_match_prefix(candidate):
                self.buffers[choice_index] = candidate
                message.pop("content", None)
            else:
                message["content"] = candidate
                self.resolved.add(choice_index)
                self.buffers.pop(choice_index, None)
                return

        if (
            choice.get("finish_reason") is not None
            and choice_index in self.buffers
        ):
            message["content"] = self.buffers.pop(choice_index)
            self.resolved.add(choice_index)


def _could_match_prefix(text: str) -> bool:
    """Check if text can still grow into a full regex match."""
    s = text.lstrip().lower()

    if len(s) < len(_LANGUAGE_HEADER):
        return _LANGUAGE_HEADER.startswith(s)

    return s.startswith(_LANGUAGE_HEADER) and len(s) <= _MAX_PREFIX_LEN


def _parse_language_prefix(text: str) -> tuple[str, str] | None:
    if not (match := _LANGUAGE_PREFIX_RE.match(text)):
        return None

    language = match.group(1).strip().capitalize()
    return language, match.group(2)


def _append_language_stage(
    message: dict,
    language: str,
    *,
    streaming: bool,
) -> None:
    cc = message.setdefault("custom_content", {})
    stages = cc.setdefault("stages", [])

    stage: dict[str, object] = {
        "name": f"Language: {language}",
        "status": "completed",
    }
    if streaming:
        stage["index"] = 0

    stages.append(stage)


_T = TypeVar("_T", bound=AsyncIterator[dict] | dict)


def extract_qwen3_asr_language(response: _T) -> _T:
    """Extract Qwen3-ASR language prefix into DIAL stage for vLLM responses."""
    if isinstance(response, dict):
        return _Qwen3AsrResponseTransformer(streaming=False)(response)

    return map_stream(_Qwen3AsrResponseTransformer(streaming=True), response)
