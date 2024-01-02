import json
import mimetypes
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple, cast

import aiohttp
import tiktoken
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.gpt4_vision.messages import (
    ImageDetail,
    create_image_message,
    create_text_message,
)
from aidial_adapter_openai.gpt4_vision.tokenization import tokenize_image
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.image_data_url import ImageDataURL
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.sse_stream import (
    format_chunk,
    parse_openai_sse_stream,
)
from aidial_adapter_openai.utils.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    create_predefined_response,
    prepend_to_async_iterator,
)
from aidial_adapter_openai.utils.tokens import calculate_prompt_tokens

# The built-in default max_tokens is 16 tokens,
# which is too small for most image-to-text use cases.
DEFAULT_MAX_TOKENS = 128

# Officially supported image types by GPT-4 Vision
SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
SUPPORTED_FILE_EXTS = ["jpg", "jpeg", "png", "webp", "gif"]

USAGE = f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them in the same message.
Only the last message in dialogue is taken into account.

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
        stream = prepend_to_async_iterator(first_chunk, stream)

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


def get_finish_reason(choice: dict) -> Optional[str]:
    """Convert GPT4 Vision finish reason to the vanilla GPT4 finish reason"""

    finish_type: Optional[str] = choice.get("finish_details", {}).get("type")

    match finish_type:
        case None:
            return None
        case "stop":
            return "stop"
        case "max_tokens":
            return "length"
        case "content_filter":
            return "content_filter"
        case _:
            logger.warning(f"Unknown finish type: {finish_type}.")
            return None


async def generate_stream(stream: AsyncIterator[dict]) -> AsyncIterator[Any]:
    is_stream = True

    id = ""
    created = ""
    finish_reason = "stop"
    usage = None

    yield format_chunk(
        build_chunk(id, None, {"role": "assistant"}, created, is_stream)
    )

    async for chunk in stream:
        logger.debug(f"chunk: {json.dumps(chunk)}")

        if "error" in chunk:
            yield chunk
            yield END_CHUNK
            return

        id = chunk.get("id", id)
        created = chunk.get("created", created)
        usage = chunk.get("usage", usage)

        choice: dict = chunk.get("choices", [{}])[0]

        finish_reason = get_finish_reason(choice) or finish_reason
        content = choice.get("delta", {}).get("content")

        if content is None:
            continue

        # TODO: Accumulate content and then compute usage
        yield format_chunk(
            build_chunk(id, None, {"content": content}, created, is_stream)
        )

    yield format_chunk(
        build_chunk(id, finish_reason, {}, created, is_stream, usage=usage)
    )
    yield END_CHUNK


def guess_image_type(attachment: Any) -> Optional[str]:
    type = attachment.get("type")
    if type is None:
        return None

    if type in SUPPORTED_IMAGE_TYPES:
        return type

    if "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        url = attachment.get("url")
        if url is None:
            return None

        file_type = mimetypes.guess_type(url)[0]

        if file_type in SUPPORTED_IMAGE_TYPES:
            return file_type

    return None


async def download_image(
    file_storage: Optional[FileStorage], attachment: Any
) -> Optional[ImageDataURL]:
    type = guess_image_type(attachment)
    if type is None:
        return None

    if "data" in attachment:
        return ImageDataURL(type=type, data=attachment["data"])

    if "url" in attachment:
        attachment_link: str = attachment["url"]

        image_url = ImageDataURL.from_data_url(attachment_link)
        if image_url is not None:
            if image_url.type in SUPPORTED_IMAGE_TYPES:
                return image_url
            else:
                return None

        try:
            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                data = await file_storage.download_file_as_base64(url)
            else:
                data = await download_file_as_base64(attachment_link)

            return ImageDataURL(type=type, data=data)
        except Exception:
            logger.warning("Failed to download image from URL")
            return None

    return None


async def transform_message(
    file_storage: Optional[FileStorage], message: dict
) -> Tuple[dict, int] | str:
    content = message.get("content", "")
    custom_content = message.get("custom_content", {})
    attachments = custom_content.get("attachments", [])

    message = {k: v for k, v in message.items() if k != "custom_content"}

    if len(attachments) == 0:
        return message, 0

    logger.debug(f"original attachments: {attachments}")

    download_results: List[Optional[ImageDataURL]] = [
        await download_image(file_storage, attachment)
        for attachment in attachments
    ]

    logger.debug(f"download results: {download_results}")

    image_urls: List[ImageDataURL] = [
        im for im in download_results if im is not None
    ]

    detail: ImageDetail = "auto"

    image_messages: List[dict] = [
        create_image_message(image, detail) for image in image_urls
    ]

    image_tokens: List[int] = [tokenize_image(im, detail) for im in image_urls]
    total_image_tokens = sum(image_tokens)

    logger.debug(f"image tokens: {image_tokens}")

    if len(image_urls) < len(attachments):
        return (
            "Some of the attachments are either invalid or failed to download."
            + "\n\n"
            + USAGE
        )

    sub_messages: List[dict] = [create_text_message(content)] + image_messages

    return {**message, "content": sub_messages}, total_image_tokens


async def transform_messages(
    file_storage: Optional[FileStorage], messages: List[dict]
) -> Tuple[List[dict], int] | str:
    last_message = messages[-1]
    result = await transform_message(file_storage, last_message)

    if isinstance(result, str):
        return result

    new_message, total_image_tokens = result

    return [new_message], total_image_tokens


async def chat_completion(
    request: Any,
    upstream_endpoint: str,
    api_key: str,
    is_stream: bool,
    file_storage: Optional[FileStorage],
) -> Response:
    if request.get("n", 1) > 1:
        raise HTTPException(
            status_code=422,
            message="The deployment doesn't support n > 1",
            type="invalid_request_error",
        )

    messages: List[Any] = request["messages"]
    if len(messages) == 0:
        raise HTTPException(
            status_code=422,
            message="The request doesn't contain any messages",
            type="invalid_request_error",
        )

    api_url = upstream_endpoint + "?api-version=2023-12-01-preview"

    result = await transform_messages(file_storage, messages)

    if isinstance(result, str):
        return create_predefined_response(result, is_stream)

    new_messages, prompt_image_tokens = result

    model = "gpt-4"
    encoding = tiktoken.encoding_for_model(model)
    prompt_text_tokens = calculate_prompt_tokens(messages, model, encoding)
    prompt_tokens = prompt_text_tokens + prompt_image_tokens

    max_tokens = request.get("max_tokens", DEFAULT_MAX_TOKENS)

    request = {
        **request,
        "max_tokens": max_tokens,
        "messages": new_messages,
    }

    headers = {"api-key": api_key}

    if is_stream:
        response = await predict_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        return StreamingResponse(
            generate_stream(parse_openai_sse_stream(response)),
            media_type="text/event-stream",
        )
    else:
        response = await predict_non_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        id = response["id"]
        created = response["created"]
        content = response["choices"][0]["message"].get("content", "")
        usage = response["usage"]

        actual_prompt_tokens = usage["prompt_tokens"]
        if actual_prompt_tokens != prompt_tokens:
            logger.debug(
                f"Estimated prompt tokens ({prompt_tokens}) don't match the actual ones ({actual_prompt_tokens})"
            )

        finish_reason = get_finish_reason(response) or "stop"

        return JSONResponse(
            content=build_chunk(
                id,
                finish_reason,
                {"role": "assistant", "content": content},
                created,
                is_stream,
                usage=usage,
            )
        )
