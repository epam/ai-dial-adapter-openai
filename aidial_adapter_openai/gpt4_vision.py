import json
from typing import Any, AsyncIterator, Dict, Literal, Optional, TypedDict, cast

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import (
    FileStorage,
    attachment_link_to_absolute_url,
    download_file_as_base64,
)
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
    parse_sse_stream,
    prepend_to_async_iterator,
)

# The built-in default max_tokens is 16 tokens,
# which is too small for most image-to-text use cases.
DEFAULT_MAX_TOKENS = 128


class ImageUrl(TypedDict, total=False):
    url: str
    detail: Optional[Literal["high", "low"]]


class ImageSubmessage(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


def create_image_message(url: str) -> ImageSubmessage:
    return {"type": "image_url", "image_url": {"url": url}}


def base64_to_image_url(type: str, base64_image: str) -> str:
    return f"data:{type};base64,{base64_image}"


async def transpose_stream(
    stream: AsyncIterator[bytes | JSONResponse],
) -> AsyncIterator[bytes] | JSONResponse:
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
) -> AsyncIterator[bytes | JSONResponse]:
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


async def generate_stream(stream: AsyncIterator[dict]) -> AsyncIterator[Any]:
    is_stream = True

    id = ""
    created = ""
    finish_reason = "stop"

    yield chunk_format(
        build_chunk(id, None, {"role": "assistant"}, created, is_stream)
    )

    async for chunk in stream:
        logger.debug(f"chunk: {json.dumps(chunk)}")

        if "error" in chunk:
            yield chunk
            yield END_CHUNK
            return

        if "id" in chunk:
            id = chunk["id"]

        if "created" in chunk:
            created = chunk["created"]

        choice: dict = (chunk.get("choices") or [{}])[0]

        finish_type: Optional[str] = (choice.get("finish_details") or {}).get(
            "type"
        )

        match finish_type:
            case None:
                pass
            case "stop":
                finish_reason = "stop"
            case "max_tokens":
                finish_reason = "length"
            case "content_filter":
                finish_reason = "content_filter"
            case _:
                logger.warning(
                    f"Unknown finish type: {finish_type}. Defaulting to stop"
                )

        content = choice.get("delta", {}).get("content")

        if content is None:
            continue

        yield chunk_format(
            build_chunk(id, None, {"content": content}, created, is_stream)
        )

    yield chunk_format(build_chunk(id, finish_reason, {}, created, is_stream))
    yield END_CHUNK


async def transform_attachment(
    file_storage: Optional[FileStorage], attachment: Any
) -> Optional[ImageSubmessage]:
    type = attachment.get("type")
    if type is None or not type.startswith("image/"):
        return None

    if "data" in attachment:
        attachment_link: str = base64_to_image_url(type, attachment["data"])
        return create_image_message(attachment_link)

    if "url" in attachment:
        attachment_link: str = attachment["url"]

        try:
            url = attachment_link_to_absolute_url(file_storage, attachment_link)
            base64_str = await download_file_as_base64(file_storage, url)
            image_url = base64_to_image_url(type, base64_str)
            return create_image_message(image_url)
        except Exception:
            logger.warning(
                f"Failed to download image from URL: {attachment_link}"
            )
            return create_image_message(attachment_link)

    return None


async def transform_message(
    file_storage: Optional[FileStorage], message: Any
) -> Any:
    content = message.get("content")
    custom_content = message.get("custom_content")

    if content is None or custom_content is None:
        return message

    message = {k: v for k, v in message.items() if k != "custom_content"}

    attachments = custom_content.get("attachments")
    if attachments is None or not isinstance(attachments, list):
        return message

    logger.debug(f"original attachments: {attachments}")

    new_attachments = [
        await transform_attachment(file_storage, attachment)
        for attachment in attachments
    ]
    new_attachments = [m for m in new_attachments if m is not None]

    logger.debug(f"transformed attachments: {str(new_attachments)[:100]}")

    new_content = [{"type": "text", "text": content}] + new_attachments

    return {**message, "content": new_content}


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

    api_url = upstream_endpoint + "?api-version=2023-12-01-preview"

    max_tokens = request.get("max_tokens", DEFAULT_MAX_TOKENS)

    new_messages = [
        await transform_message(file_storage, message)
        for message in request["messages"][-1:]
    ]

    request = {
        **request,
        "max_tokens": max_tokens,
        "messages": new_messages,
    }

    headers = {"api-key": api_key}

    if is_stream:
        response = await transpose_stream(
            predict_stream(api_url, headers, request)
        )
        if isinstance(response, Response):
            return response

        return StreamingResponse(
            generate_stream(parse_sse_stream(response)),
            media_type="text/event-stream",
        )
    else:
        response = await predict_non_stream(api_url, headers, request)
        if isinstance(response, Response):
            return response

        id = response["id"]
        created = response["created"]
        content = response["choices"][0]["message"].get("content") or ""
        usage = response["usage"]

        return JSONResponse(
            content=build_chunk(
                id,
                "stop",
                {"role": "assistant", "content": content},
                created,
                False,
                usage=usage,
            )
        )
