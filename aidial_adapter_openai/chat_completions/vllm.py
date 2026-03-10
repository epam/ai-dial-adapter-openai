from typing import AsyncIterator, List, Set, TypeVar

from aidial_sdk.exceptions import InvalidRequestError
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import BaseModel

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    chunk_to_dict,
    debug_print,
    map_stream,
)
from aidial_adapter_openai.utils.truncate_prompt import DiscardedMessages
from aidial_adapter_openai.utils.vllm_tokenizer import VllmTokenizer


def _extract_max_prompt_tokens(request: dict) -> int | None:
    if (max_prompt_tokens := request.pop("max_prompt_tokens", None)) is None:
        return None

    if not isinstance(max_prompt_tokens, int):
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is not of type 'integer'",
            param="max_prompt_tokens",
        )

    if max_prompt_tokens < 1:
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is less than the minimum of 1",
            param="max_prompt_tokens",
        )

    return max_prompt_tokens


async def _truncate_messages(
    request: dict, messages: List[MultiModalMessage], tokenizer: VllmTokenizer
) -> tuple[List[MultiModalMessage], DiscardedMessages | None]:
    """vLLM truncation calls tokenize with the full message list each pass."""
    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is None:
        return messages, None

    (
        messages,
        discarded_indices,
        prompt_tokens,
    ) = await tokenizer.truncate_prompt(
        original_request=request,
        messages=messages,
        max_prompt_tokens=max_prompt_tokens,
    )

    logger.debug(
        f"vLLM estimated prompt tokens after truncation: {prompt_tokens}, "
        f"discarded messages indices: {discarded_indices}"
    )

    return messages, discarded_indices


async def chat_completion(
    *,
    request: dict,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: VllmTokenizer,
    eliminate_empty_choices: bool,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    messages: List[dict] = request["messages"]

    multi_modal_messages = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)

    (
        multi_modal_messages,
        discarded_messages,
    ) = await _truncate_messages(request, multi_modal_messages, tokenizer)

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    # vLLM includes usage in stream mode only if stream_options.include_usage=true.
    if request.get("stream"):
        stream_options = request.get("stream_options")
        if not isinstance(stream_options, dict):
            stream_options = {}
        stream_options["include_usage"] = True
        request["stream_options"] = stream_options

    response: (
        AsyncStream[ChatCompletionChunk] | ChatCompletion
    ) = await call_with_extra_body(client.chat.completions.create, request)

    if isinstance(response, AsyncStream):
        body = _generate_stream(
            stream=map_stream(chunk_to_dict, response),
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )
        return ResponseWithHeaders(headers=None, body=body)

    if eliminate_empty_choices and response.choices:
        response.choices = [c for c in response.choices if c]

    data = response.to_dict()
    if discarded_messages is not None:
        data |= {"statistics": {"discarded_messages": discarded_messages}}

    debug_print("response", data)
    return ResponseWithHeaders(headers=None, body=data)


async def _generate_stream(
    *,
    stream: AsyncIterator[dict],
    discarded_messages: DiscardedMessages | None,
    eliminate_empty_choices: bool,
) -> AsyncIterator[dict]:
    """Pass through vLLM chunks and append discarded message statistics to the last chunk."""
    last_chunk = None

    async for chunk in stream:
        if eliminate_empty_choices and isinstance(chunk.get("choices"), list):
            chunk["choices"] = [c for c in chunk["choices"] if c]

        if last_chunk is not None:
            yield last_chunk
        last_chunk = chunk

    if last_chunk is not None:
        if discarded_messages is not None:
            last_chunk["statistics"] = {
                "discarded_messages": discarded_messages
            }
        yield last_chunk


class _ReasoningResponseTransformer(BaseModel):
    """Extracts reasoning from vLLM responses into DIAL Stages.

    vLLM reasoning models (e.g. DeepSeek-R1) return reasoning tokens
    in a ``reasoning`` field on both the message (non-streaming) and
    the delta (streaming).

    See:
    - Non-streaming: https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_with_reasoning/
    - Streaming: https://docs.vllm.ai/en/latest/examples/online_serving/openai_chat_completion_with_reasoning_streaming/
    """

    opened_reasoning_stages: Set[int] = set()

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
