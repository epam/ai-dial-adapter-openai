from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import Response

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.gpt4_multi_modal.transformation import (
    SUPPORTED_FILE_EXTS,
    ResourceProcessor,
)
from aidial_adapter_openai.utils.aiohttp.client import post, post2
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.caching import (
    get_prompt_tokens_from_response,
    get_response_headers_for_caching,
)
from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionBlock,
)
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage
from aidial_adapter_openai.utils.parsers import chat_completions_parser
from aidial_adapter_openai.utils.sse_stream import parse_openai_sse_stream
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
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
            # Exhaust the stream
            async for _ in stream:
                pass
            return chunk
        else:
            first_chunk = chunk
            break

    stream = cast(AsyncIterator[bytes], stream)
    if first_chunk is not None:
        stream = prepend_to_stream(first_chunk, stream)

    return stream


async def predict_stream(
    url: str, headers: Dict[str, str], request: Any
) -> AsyncIterator[bytes] | Response:
    return await transpose_stream(predict_stream_raw2(url, headers, request))


async def predict_stream_raw2(
    url: str, headers: Dict[str, str], request: Any
) -> AsyncIterator[bytes | Response]:
    async with post2(url, headers, request) as response:
        if not response.is_success:
            yield Response(
                status_code=response.status_code,
                content=response.content,
            )

        async for line in response.aiter_lines():
            yield line.encode()


async def predict_stream_raw(
    url: str, headers: Dict[str, str], request: Any
) -> AsyncIterator[bytes | Response]:
    async with post(url, headers, request) as response:
        if not response.ok:
            yield Response(
                status_code=response.status,
                content=await response.content.read(),
            )

        async for line in response.content:
            yield line


async def predict_non_stream(
    url: str, headers: Dict[str, str], request: Any
) -> dict | Response:
    async with post(url, headers, request) as response:
        if not response.ok:
            return Response(
                status_code=response.status,
                content=await response.content.read(),
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
    request_headers: Mapping[str, str],
    deployment: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
    tokenizer: MultiModalTokenizer,
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
        "messages": [m.raw_message for m in multi_modal_messages],
    }

    openai_endpoint = chat_completions_parser.parse(upstream_endpoint)
    headers = openai_endpoint.get_auth_headers(creds)

    if is_stream:
        response = await predict_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        T = TypeVar("T")

        def debug_print(chunk: T) -> T:
            logger.debug(f"chunk: {chunk}")
            return chunk

        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: estimated_prompt_tokens,
        )

        body = map_stream(
            debug_print,
            generate_stream(
                stream=parse_openai_sse_stream(response),
                get_prompt_tokens=lambda: estimated_prompt_tokens,
                tokenize_response=tokenizer.tokenize_response,
                deployment=deployment,
                discarded_messages=discarded_messages,
                eliminate_empty_choices=eliminate_empty_choices,
            ),
        )

        return ResponseWithHeaders(headers=headers, body=body)
    else:
        response = await predict_non_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

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

        headers = get_response_headers_for_caching(
            request_headers=request_headers,
            request_body=request,
            get_request_tokens=lambda: get_prompt_tokens_from_response(response)
            or estimated_prompt_tokens,
        )

        return ResponseWithHeaders(headers=headers, body=response)
