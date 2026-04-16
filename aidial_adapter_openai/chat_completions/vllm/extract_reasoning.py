from collections.abc import AsyncIterator
from typing import TypeVar

from pydantic import BaseModel

from aidial_adapter_openai.utils.streaming import map_stream


class _ReasoningResponseTransformer(BaseModel):
    """Extracts reasoning from vLLM responses into DIAL Stages.

    vLLM reasoning models (e.g. DeepSeek-R1) return reasoning tokens
    in a ``reasoning`` field on both the message (non-streaming) and
    the delta (streaming).

    See:
    - Non-streaming: https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_with_reasoning/
    - Streaming: https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_with_reasoning_streaming/
    """

    opened_reasoning_stages: set[int] = set()

    streaming: bool

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices") or []
        for choice in choices:
            choice_index = choice.get("index")
            message = choice.get(self.message_key, {})
            reasoning = message.pop("reasoning", None)

            is_ongoing = reasoning is not None
            is_opening = (
                choice_index not in self.opened_reasoning_stages and is_ongoing
            )
            is_closing = (
                choice_index in self.opened_reasoning_stages
                and choice.get("finish_reason") is not None
            )

            if is_opening:
                self.opened_reasoning_stages.add(choice_index)

            if is_opening or is_ongoing or is_closing:
                cc = message.setdefault("custom_content", {})
                stages = cc.setdefault("stages", [])

                opening_fields = {"name": "Reasoning"} if is_opening else {}
                closing_fields = {"status": "completed"} if is_closing else {}
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = {"content": reasoning} if is_ongoing else {}

                stages.append(
                    {
                        **content_fields,
                        **streaming_fields,
                        **opening_fields,
                        **closing_fields,
                    }
                )

        return chunk


_T = TypeVar("_T", bound=AsyncIterator[dict] | dict)


def extract_reasoning(response: _T) -> _T:
    """Extract vLLM ``reasoning`` field into DIAL Stages.

    Handles both streaming (``delta.reasoning``) and
    non-streaming (``message.reasoning``) response formats.
    """
    if isinstance(response, dict):
        return _ReasoningResponseTransformer(streaming=False)(response)
    else:
        return map_stream(
            _ReasoningResponseTransformer(streaming=True), response
        )
