from collections.abc import AsyncIterator
from typing import overload

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import map_stream


def _to_dial_finish_reason(finish_reason: str | None) -> str | None:
    if finish_reason in ("model_length", "error"):
        return "length"
    return finish_reason


class _MistralResponseTransformer:
    def __init__(self, streaming: bool):
        self.opened_reasoning_stages: set[int] = set()
        self.streaming = streaming

    @property
    def message_key(self) -> str:
        return "delta" if self.streaming else "message"

    def __call__(self, chunk: dict) -> dict:
        choices = chunk.get("choices") or []
        for choice in choices:
            choice_index = choice.get("index")
            message = choice.get(self.message_key, {})
            content = message.get("content")

            text_parts: list[str] = []
            thinking_parts: list[str] = []

            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    item_type = item.get("type")
                    if item_type == "text":
                        text_parts.append(item.get("text") or "")
                    elif item_type == "thinking":
                        thinking_value = item.get("thinking")
                        if not isinstance(thinking_value, list):
                            continue
                        for thinking_item in thinking_value:
                            if not isinstance(thinking_item, dict):
                                continue
                            thinking_type = thinking_item.get("type")
                            if thinking_type == "text":
                                thinking_parts.append(
                                    thinking_item.get("text") or ""
                                )
                            else:
                                logger.warning(
                                    f"The response thinking part of type {thinking_type!r} isn't supported and will be ignored"
                                )
                    else:
                        logger.warning(
                            f"The response content part of type {item_type!r} isn't supported and will be ignored"
                        )
                message["content"] = "".join(text_parts) if text_parts else None

            choice["finish_reason"] = _to_dial_finish_reason(
                choice.get("finish_reason")
            )

            thinking_text = "".join(thinking_parts)
            is_ongoing = bool(thinking_text)
            is_opening = (
                choice_index not in self.opened_reasoning_stages and is_ongoing
            )

            if is_opening:
                self.opened_reasoning_stages.add(choice_index)

            is_closing = (
                choice_index in self.opened_reasoning_stages
                and choice.get("finish_reason") is not None
            )

            if is_opening or is_ongoing or is_closing:
                cc = message.setdefault("custom_content", {})
                stages = cc.setdefault("stages", [])

                opening_fields = {"name": "Reasoning"} if is_opening else {}
                closing_fields = {"status": "completed"} if is_closing else {}
                streaming_fields = {"index": 0} if self.streaming else {}
                content_fields = (
                    {"content": thinking_text} if is_ongoing else {}
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


@overload
def extract_reasoning_content(response_or_stream: dict) -> dict: ...


@overload
def extract_reasoning_content(
    response_or_stream: AsyncIterator[dict],
) -> AsyncIterator[dict]: ...


def extract_reasoning_content(
    response_or_stream: dict | AsyncIterator[dict],
) -> dict | AsyncIterator[dict]:
    if isinstance(response_or_stream, dict):
        return _MistralResponseTransformer(streaming=False)(response_or_stream)
    else:
        return map_stream(
            _MistralResponseTransformer(streaming=True), response_or_stream
        )


def _response_to_dict(obj: ChatCompletionChunk | ChatCompletion) -> dict:
    # Magistral returns an invalid OpenAI response.
    # Suppressing warnings from pydantic:
    # > Expected `str` but got `list` with value `[{'type': 'thinking', 'thinking': []}]` - serialized value may not be as expected
    return obj.to_dict(warnings=False)


async def chat_completion(
    *, request: dict, client: AsyncAzureOpenAI | AsyncOpenAI
) -> AsyncIterator[dict] | dict:
    response: (
        AsyncStream[ChatCompletionChunk] | ChatCompletion
    ) = await call_with_extra_body(client.chat.completions.create, request)

    if isinstance(response, AsyncStream):
        raw_stream = map_stream(_response_to_dict, response)
        return extract_reasoning_content(raw_stream)
    else:
        return extract_reasoning_content(_response_to_dict(response))
