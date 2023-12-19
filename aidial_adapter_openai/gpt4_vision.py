import json
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    cast,
)

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import FileStorage
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
    file_storage: Optional[FileStorage], attachment: Any, download_image: bool
) -> Optional[ImageSubmessage]:
    if "type" not in attachment:
        return None

    type = attachment["type"]

    if not type.startswith("image/"):
        return None

    if "data" in attachment:
        url: str = base64_to_image_url(type, attachment["data"])
        return create_image_message(url)

    if "url" in attachment:
        url: str = attachment["url"]

        if not download_image:
            return create_image_message(url)

        if file_storage is None:
            logger.warning(
                f"File storage is not initialized. Cannot download image from URL: {url}"
            )
            return create_image_message(url)

        absolute_url = file_storage.attachment_link_to_absolute_url(url)
        base64_str = await file_storage.download_file_as_base64(absolute_url)
        image_url = base64_to_image_url(type, base64_str)

        return create_image_message(image_url)

    return None


async def transform_message(
    file_storage: Optional[FileStorage], message: Any, download_image: bool
) -> Any:
    if "content" not in message:
        return message

    content = message["content"]

    if "custom_content" not in message:
        return message

    custom_content = message["custom_content"]

    message = {k: v for k, v in message.items() if k != "custom_content"}

    if "attachments" not in custom_content:
        return message

    attachments = custom_content["attachments"]

    if not isinstance(attachments, list):
        return message

    sub_messages = [
        await transform_attachment(file_storage, attachment, download_image)
        for attachment in attachments
    ]
    sub_messages = [m for m in sub_messages if m is not None]

    if len(sub_messages) == 0:
        return message

    new_content = [{"type": "text", "text": content}] + sub_messages
    return {**message, "content": new_content}


async def transform_messages(
    messages: List[Any], file_storage: Optional[FileStorage]
) -> List[Any]:
    ret: List[Any] = []

    for idx, message in enumerate(messages):
        is_last = idx == len(messages) - 1
        ret.append(
            await transform_message(
                message=message,
                download_image=is_last,
                file_storage=file_storage,
            )
        )

    return ret


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

    request = {
        **request,
        "max_tokens": max_tokens,
        "messages": await transform_messages(request["messages"], file_storage),
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
