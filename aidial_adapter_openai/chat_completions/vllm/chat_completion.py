from collections.abc import AsyncIterator

from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm.audio_transformer import (
    transform_audio,
)
from aidial_adapter_openai.chat_completions.vllm.tokenizer import (
    VllmTokenizer,
)
from aidial_adapter_openai.dial_api.request import extract_max_prompt_tokens
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    add_statistics_to_response,
    chunk_to_dict,
    debug_print,
    map_stream,
)
from aidial_adapter_openai.utils.truncate_prompt import (
    truncate_prompt,
)
from aidial_adapter_openai.utils.truncation_types import (
    DiscardedMessages,
    TruncatedTokens,
)


async def _transform_messages(
    messages: list[dict],
    file_storage: FileStorage | None,
) -> list[MultiModalMessage]:
    multi_modal_messages = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)
    return transform_audio(multi_modal_messages)


async def transform_vllm_messages(
    messages: list[dict],
    file_storage: FileStorage | None,
) -> list[dict]:
    """Apply the DIAL-to-OpenAI multimodal transformation expected by vLLM."""
    return [
        m.raw_message for m in await _transform_messages(messages, file_storage)
    ]


async def truncate_vllm_prompt(
    *,
    request: dict,
    file_storage: FileStorage | None,
    max_prompt_tokens: int,
    tokenizer: VllmTokenizer,
) -> tuple[list[dict], DiscardedMessages, TruncatedTokens]:
    transformed = await transform_vllm_messages(
        request["messages"], file_storage
    )
    return await truncate_prompt(
        tokenizer=tokenizer,
        original_request=request,
        messages=transformed,
        get_raw_message=lambda m: m,
        max_prompt_tokens=max_prompt_tokens,
    )


async def chat_completion(
    *,
    request: dict,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: VllmTokenizer,
) -> ResponseWithHeaders[AsyncIterator[dict] | dict]:
    max_prompt_tokens = extract_max_prompt_tokens(request)
    discarded_messages: DiscardedMessages | None

    if max_prompt_tokens is not None:
        (
            messages,
            discarded_messages,
            prompt_tokens,
        ) = await truncate_vllm_prompt(
            request=request,
            file_storage=file_storage,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=tokenizer,
        )
        logger.debug(
            f"vLLM estimated prompt tokens after truncation: {prompt_tokens}, "
            f"discarded messages indices: {discarded_messages}"
        )
    else:
        messages = await transform_vllm_messages(
            request["messages"], file_storage
        )
        discarded_messages = None

    request["messages"] = messages

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
        body = add_statistics_to_response(
            map_stream(chunk_to_dict, response),
            discarded_messages=discarded_messages,
        )
        return ResponseWithHeaders(headers=None, body=body)

    data = response.to_dict()
    add_statistics_to_response(data, discarded_messages=discarded_messages)

    debug_print("response", data)
    return ResponseWithHeaders(headers=None, body=data)
