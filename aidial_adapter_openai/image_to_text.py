import base64
import os
from typing import Any, AsyncGenerator, List, Literal, Optional, TypedDict

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.attachment_link import AttachmentLink
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
)

DIAL_URL = os.getenv("DIAL_URL")
if not DIAL_URL:
    raise ValueError(
        "DIAL_URL environment variables must be initialized to use GPT-4 with Vision"
    )


class ImageUrl(TypedDict, total=False):
    url: str
    detail: Optional[Literal["high", "low"]]


class ImageSubmessage(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


def create_image_message(url: str) -> ImageSubmessage:
    return {"type": "image_url", "image_url": {"url": url}}


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


async def download_file(
    jwt: str, attachment_link: AttachmentLink
) -> Optional[bytes]:
    url = attachment_link.absolute_url
    headers = {"authorization": jwt}

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            if not response.ok:
                logger.warning(
                    f"Failed to load index from {url}: {response.status}, {response.reason}"
                )
                return None
            return await response.read()


def base64_to_image_url(type: str, base64_image: str) -> str:
    return f"data:{type};base64,{base64_image}"


async def download_base64_image(
    jwt: str, type: str, attachment_link: AttachmentLink
) -> Optional[str]:
    bytes = await download_file(jwt, attachment_link)
    if bytes is None:
        return None

    text = base64.b64encode(bytes).decode("ascii")
    return base64_to_image_url(type, text)


async def transform_attachment(
    jwt: str, attachment: Any, download_image: bool
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
            create_image_message(url)

        link = AttachmentLink.from_url_or_path(DIAL_URL, url)

        image_url = await download_base64_image(jwt, type, link)
        return create_image_message(image_url or url)

    return None


async def transform_message(
    jwt: str, message: Any, download_image: bool
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
        await transform_attachment(jwt, attachment, download_image)
        for attachment in attachments
    ]
    sub_messages = [m for m in sub_messages if m is not None]

    if len(sub_messages) == 0:
        return message

    new_content = [{"type": "text", "text": content}] + sub_messages
    return {**message, "content": new_content}


async def transform_messages(jwt: str, messages: List[Any]) -> List[Any]:
    ret: List[Any] = []

    for idx, message in enumerate(messages):
        is_last = idx == len(messages) - 1
        ret.append(
            await transform_message(
                jwt=jwt, message=message, download_image=is_last
            )
        )

    return ret


async def image_to_text_chat_completion(
    request: Any,
    upstream_endpoint: str,
    api_key: str,
    jwt: str,
    is_stream: bool,
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
        "messages": await transform_messages(jwt, request["messages"]),
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
