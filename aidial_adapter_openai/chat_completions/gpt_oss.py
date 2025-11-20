"""
Azure OpenAI documentation doesn't specify exactly how
OpenAI Harmony Response Format (which is used by gpt-oss under the hood)
is mapped unto the Chat Completions API.

Reversed engineering and testing shows that reasoning content is provided
in `message.reasoning_content` field of the response.

The module provides helpers that add the reasoning content
into a dedicated DIAL Stage, so that thought process
becomes visible to the DIAL Chat UI.

Besides that there is no visible deviation of the Chat Completions API.

Relevant links:
https://cookbook.openai.com/articles/openai-harmony
https://cookbook.openai.com/articles/gpt-oss/verifying-implementations
https://cookbook.openai.com/articles/gpt-oss/handle-raw-cot
"""

from typing import AsyncIterator, Set, TypeVar

from pydantic import BaseModel

from aidial_adapter_openai.utils.streaming import map_stream


class _ResponseTransformer(BaseModel):
    opened_reasoning_stages: Set[int] = set()
    """Indices of choices where a reasoning stage was open.
    """

    streaming: bool

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices") or []
        for choice in choices:
            choice_index = choice.get("index")
            message = choice.get(self.message_key, {})
            reasoning_content = message.pop("reasoning_content", None)

            is_ongoing = reasoning_content is not None
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
                # Limitation: any existing "stages" are ignored,
                # whereas their indices should be offset by 1.
                # However, we don't expect any stages from the upstream.
                cc = message.setdefault("custom_content", {})
                stages = cc.setdefault("stages", [])

                opening_fields = {"name": "Reasoning"} if is_opening else {}
                closing_fields = {"status": "completed"} if is_closing else {}
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = (
                    {"content": reasoning_content} if is_ongoing else {}
                )

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


def extract_reasoning_tokens(response: _T) -> _T:
    if isinstance(response, dict):
        return _ResponseTransformer(streaming=False)(response)
    else:
        return map_stream(_ResponseTransformer(streaming=True), response)
