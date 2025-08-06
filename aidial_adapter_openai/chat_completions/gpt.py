from typing import Any, List, Mapping, Tuple

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError, RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types.chat import ChatCompletion, ChatCompletionChunk

from aidial_adapter_openai.chat_completions.input import SupportedInputs
from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.caching import get_response_headers_for_caching
from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionBlock,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
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
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
    truncate_prompt,
)


def multi_modal_truncate_prompt(
    request: dict,
    messages: List[MultiModalMessage],
    max_prompt_tokens: int,
    tokenizer: Tokenizer,
) -> Tuple[List[MultiModalMessage], DiscardedMessages, TruncatedTokens]:
    return truncate_prompt(
        messages=messages,
        message_tokens=tokenizer.tokenize_request_message,
        is_system_message=lambda message: message.raw_message["role"]
        == "system",
        max_prompt_tokens=max_prompt_tokens,
        initial_prompt_tokens=tokenizer.tokenize_request(request, []),
    )


def _validate_request(request: Any) -> List[Any]:
    errors: List[str] = []

    if (n := request.get("n")) not in [None, 1]:
        errors.append(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    if not (messages := request.get("messages")):
        errors.append("The request doesn't contain any messages.")

    if errors:
        raise RequestValidationError(" ".join(errors))

    return messages


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


async def chat_completion(
    request: dict,
    model_name: str,
    request_headers: Mapping[str, str],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    supported_inputs: SupportedInputs,
    tokenizer: Tokenizer,
    eliminate_empty_choices: bool,
):
    messages = _validate_request(request)

    transform_result = await ResourceProcessor(
        file_storage=file_storage,
        supported_image_types=supported_inputs.input_types,
    ).transform_messages(messages)

    if isinstance(transform_result, DialException):
        logger.error(f"Failed to prepare request: {transform_result.message}")
        is_stream = bool(request.get("stream"))
        chunk = None
        if (usage := supported_inputs.usage_message) is not None:
            chunk = create_stage_chunk(
                model_name=model_name,
                stage_title="Usage",
                stage_content=usage,
                stream=is_stream,
            )
        return create_response_from_chunk(
            chunk=chunk, exc=transform_result, stream=is_stream
        )

    multi_modal_messages = transform_result
    discarded_messages = None

    if (max_prompt_tokens := _extract_max_prompt_tokens(request)) is not None:
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

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, request)
    )

    if isinstance(response, AsyncStream):
        response_headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: estimated_prompt_tokens,
        )

        body = generate_stream(
            stream=map_stream(chunk_to_dict, response),
            get_prompt_tokens=lambda: estimated_prompt_tokens,
            tokenize_response=tokenizer.tokenize_response,
            model=model_name,
            discarded_messages=discarded_messages,
            eliminate_empty_choices=eliminate_empty_choices,
        )

        return ResponseWithHeaders(headers=response_headers, body=body)
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

        response_headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: actual_prompt_tokens
            or estimated_prompt_tokens,
        )

        return ResponseWithHeaders(headers=response_headers, body=body)
