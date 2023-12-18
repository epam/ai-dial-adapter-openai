from typing import Any, AsyncGenerator, List, Literal, Optional, TypedDict

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.storage import FileStorage
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
)


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


async def predict(api_url: str, api_key: str, request: Any) -> Any:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            json=request,
            headers={"api-key": api_key},
        ) as response:
            status_code = response.status

            data = await response.json()

            if status_code == 200:
                return data
            else:
                return JSONResponse(content=data, status_code=status_code)


async def generate_stream(
    id: str, created: str, content: dict, usage: dict
) -> AsyncGenerator[Any, Any]:
    yield chunk_format(
        build_chunk(id, None, {"role": "assistant"}, created, True)
    )

    yield chunk_format(build_chunk(id, None, content, created, True))
    yield chunk_format(build_chunk(id, "stop", {}, created, True, usage=usage))

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
    request = {
        **request,
        "messages": await transform_messages(request["messages"], file_storage),
    }

    response = await predict(api_url, api_key, request)
    if isinstance(response, JSONResponse):
        return response

    usage = response["usage"]
    content = response["choices"][0]["message"]["content"]

    id = response["id"]
    created = response["created"]

    if not is_stream:
        return JSONResponse(
            content=build_chunk(
                id, "stop", {"content": content}, created, False, usage=usage
            )
        )
    else:
        return StreamingResponse(
            generate_stream(id, created, {"content": content}, usage),
            media_type="text/event-stream",
        )
