from typing import Any, List, Mapping, Optional, Tuple

from aidial_sdk.exceptions import HTTPException as DialException
from openai import AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    SUPPORTED_FILE_EXTS,
    ResourceProcessor,
)
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.caching import get_response_headers_for_caching
from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionBlock,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    chunk_to_dict,
    create_response_from_chunk,
    create_stage_chunk,
    debug_print,
    generate_stream,
    map_stream,
)
from aidial_adapter_openai.utils.tokenizer import MultiModalTokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
    truncate_prompt,
)

USAGE = f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them.

Supported image types: {', '.join(SUPPORTED_FILE_EXTS)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()


def multi_modal_truncate_prompt(
    request: dict,
    messages: List[MultiModalMessage],
    max_prompt_tokens: int,
    tokenizer: MultiModalTokenizer,
) -> Tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
    return truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=tokenizer.tokenize_request(request, []),
    )


async def gpt4o_chat_completion(
    request: Any,
    deployment: str,
    request_headers: Mapping[str, str],
    endpoint: OpenAIEndpoint | AzureOpenAIEndpoint,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
    tokenizer: MultiModalTokenizer,
    eliminate_empty_choices: bool,
):
    n: int = request.get("n") or 1
    messages: List[dict] = request.get("messages")

    transform_result = await ResourceProcessor(
        file_storage=file_storage
    ).transform_messages(messages)

    if isinstance(transform_result, DialException):
        logger.error(f"Failed to prepare request: {transform_result.message}")
        chunk = create_stage_chunk("Usage", USAGE, is_stream)
        return create_response_from_chunk(chunk, transform_result, is_stream)

    multi_modal_messages = transform_result
    discarded_messages = None
    max_prompt_tokens = request.pop("max_prompt_tokens", None)
    if max_prompt_tokens is not None:
        multi_modal_messages, discarded_messages, estimated_prompt_tokens = (
            multi_modal_truncate_prompt(
                request=request,
                messages=multi_modal_messages,
                max_prompt_tokens=max_prompt_tokens,
                tokenizer=tokenizer,
            )
        )
        logger.debug(
            f"prompt tokens after truncation: {estimated_prompt_tokens}"
        )
    else:
        estimated_prompt_tokens = tokenizer.tokenize_request(
            request, multi_modal_messages
        )
        logger.debug(
            f"prompt tokens without truncation: {estimated_prompt_tokens}"
        )

    request["messages"] = [m.raw_message for m in multi_modal_messages]

    client = endpoint.get_client({**creds, "api_version": api_version})

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncStream):
        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: estimated_prompt_tokens,
        )

        body = generate_stream(
            n=n,
            stream=map_stream(chunk_to_dict, response),
            get_prompt_tokens=lambda: estimated_prompt_tokens,
            tokenize_response=tokenizer.tokenize_response,
            deployment=deployment,
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )

        return ResponseWithHeaders(headers=headers, body=body)
    else:
        body = response.to_dict()
        if discarded_messages is not None:
            body |= {"statistics": {"discarded_messages": discarded_messages}}
        debug_print("response", body)

        actual_prompt_tokens: int | None = None
        if usage := response.usage:
            actual_prompt_tokens = usage.prompt_tokens
            if actual_prompt_tokens != estimated_prompt_tokens:
                logger.warning(
                    f"Estimated prompt tokens ({estimated_prompt_tokens}) don't match the actual ones ({actual_prompt_tokens})"
                )

            actual_completion_tokens = usage.completion_tokens
            estimated_completion_tokens = tokenizer.tokenize_response(
                ChatCompletionBlock(response=body)
            )
            if actual_completion_tokens != estimated_completion_tokens:
                logger.warning(
                    f"Estimated completion tokens ({estimated_completion_tokens}) don't match the actual ones ({actual_completion_tokens})"
                )

        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: actual_prompt_tokens
            or estimated_prompt_tokens,
        )

        return ResponseWithHeaders(headers=headers, body=body)
