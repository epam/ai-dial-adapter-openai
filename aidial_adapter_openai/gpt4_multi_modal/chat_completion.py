import os
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

import aiohttp
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.gpt4_multi_modal.gpt4_vision import (
    convert_gpt4v_to_gpt4_chunk,
)
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    SUPPORTED_FILE_EXTS,
    ResourceProcessor,
)
from aidial_adapter_openai.utils.auth import OpenAICreds, get_auth_headers
from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionBlock,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.sse_stream import parse_openai_sse_stream
from aidial_adapter_openai.utils.streaming import (
    create_response_from_chunk,
    create_stage_chunk,
    generate_stream,
    map_stream,
    prepend_to_stream,
)
from aidial_adapter_openai.utils.tokenizer import MultiModalTokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    DiscardedMessages,
    TruncatedTokens,
    truncate_prompt,
)

# The built-in default max_tokens is 16 tokens,
# which is too small for most image-to-text use cases.
GPT4V_DEFAULT_MAX_TOKENS = int(os.getenv("GPT4_VISION_MAX_TOKENS", "1024"))

USAGE = f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them.

Supported image types: {', '.join(SUPPORTED_FILE_EXTS)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()


async def transpose_stream(
    stream: AsyncIterator[bytes | Response],
) -> AsyncIterator[bytes] | Response:
    first_chunk: Optional[bytes] = None
    async for chunk in stream:
        if isinstance(chunk, Response):
            return chunk
        else:
            first_chunk = chunk
            break

    stream = cast(AsyncIterator[bytes], stream)
    if first_chunk is not None:
        stream = prepend_to_stream(first_chunk, stream)

    return stream


async def predict_stream(
    api_url: str, headers: Dict[str, str], request: Any
) -> AsyncIterator[bytes] | Response:
    return await transpose_stream(predict_stream_raw(api_url, headers, request))


async def predict_stream_raw(
    api_url: str, headers: Dict[str, str], request: Any
) -> AsyncIterator[bytes | Response]:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url, json=request, headers=headers
        ) as response:
            if response.status != 200:
                yield JSONResponse(
                    status_code=response.status, content=await response.json()
                )
                return

            async for line in response.content:
                yield line


async def predict_non_stream(
    api_url: str, headers: Dict[str, str], request: Any
) -> dict | JSONResponse:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url, json=request, headers=headers
        ) as response:
            if response.status != 200:
                return JSONResponse(
                    status_code=response.status, content=await response.json()
                )
            return await response.json()


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
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
    tokenizer: MultiModalTokenizer,
    eliminate_empty_choices: bool,
):
    return await chat_completion(
        request,
        deployment,
        upstream_endpoint,
        creds,
        is_stream,
        file_storage,
        api_version,
        tokenizer,
        lambda x: x,
        None,
        eliminate_empty_choices,
    )


async def gpt4_vision_chat_completion(
    request: Any,
    deployment: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
    tokenizer: MultiModalTokenizer,
    eliminate_empty_choices: bool,
):
    return await chat_completion(
        request,
        deployment,
        upstream_endpoint,
        creds,
        is_stream,
        file_storage,
        api_version,
        tokenizer,
        convert_gpt4v_to_gpt4_chunk,
        GPT4V_DEFAULT_MAX_TOKENS,
        eliminate_empty_choices,
    )


async def chat_completion(
    request: Any,
    deployment: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
    tokenizer: MultiModalTokenizer,
    response_transformer: Callable[[dict], dict | None],
    default_max_tokens: Optional[int],
    eliminate_empty_choices: bool,
):
    if request.get("n", 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    messages: List[Any] = request["messages"]
    if len(messages) == 0:
        raise RequestValidationError("The request doesn't contain any messages")

    api_url = f"{upstream_endpoint}?api-version={api_version}"

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

    request = {
        **request,
        "max_tokens": request.get("max_tokens") or default_max_tokens,
        "messages": [m.raw_message for m in multi_modal_messages],
    }

    headers = get_auth_headers(creds)

    if is_stream:
        response = await predict_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        T = TypeVar("T")

        def debug_print(chunk: T) -> T:
            logger.debug(f"chunk: {chunk}")
            return chunk

        return map_stream(
            debug_print,
            generate_stream(
                get_prompt_tokens=lambda: estimated_prompt_tokens,
                tokenize_response=tokenizer.tokenize_response,
                deployment=deployment,
                discarded_messages=discarded_messages,
                stream=map_stream(
                    response_transformer,
                    parse_openai_sse_stream(response),
                ),
                eliminate_empty_choices=eliminate_empty_choices,
            ),
        )
    else:
        response = await predict_non_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        response = response_transformer(response)
        if response is None:
            raise DialException(
                status_code=500,
                message="The origin returned invalid response",
                type="invalid_response_error",
            )

        if discarded_messages:
            response |= {
                "statistics": {"discarded_messages": discarded_messages}
            }

        if usage := response.get("usage"):
            actual_prompt_tokens = usage["prompt_tokens"]
            if actual_prompt_tokens != estimated_prompt_tokens:
                logger.warning(
                    f"Estimated prompt tokens ({estimated_prompt_tokens}) don't match the actual ones ({actual_prompt_tokens})"
                )

            actual_completion_tokens = usage["completion_tokens"]
            estimated_completion_tokens = tokenizer.tokenize_response(
                ChatCompletionBlock(resp=response)
            )
            if actual_completion_tokens != estimated_completion_tokens:
                logger.warning(
                    f"Estimated completion tokens ({estimated_completion_tokens}) don't match the actual ones ({actual_completion_tokens})"
                )

        return response
