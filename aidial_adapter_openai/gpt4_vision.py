import json
import re
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
from aidial_adapter_openai.utils.storage import (
    FileStorage,
    download_file_as_base64,
)
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
    create_predefined_response,
    parse_sse_stream,
    prepend_to_async_iterator,
)

# The built-in default max_tokens is 16 tokens,
# which is too small for most image-to-text use cases.
DEFAULT_MAX_TOKENS = 128

# Officially supported image types by GPT-4 Vision
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "gif"]

USAGE = f"""
### Usage

The application answers queries about attached images.
Attach images and ask questions about them in the same message.
Only the last message in dialogue is taken into account.

Supported image types: {', '.join(SUPPORTED_IMAGE_TYPES)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()


class ImageUrl(TypedDict, total=False):
    url: str
    detail: Optional[Literal["high", "low", "auto"]]


class ImageSubmessage(TypedDict):
    type: Literal["image_url"]
    image_url: ImageUrl


class TextSubmessage(TypedDict):
    type: Literal["text"]
    text: str


VisionContent = List[ImageSubmessage | TextSubmessage]


def create_image_message(url: str) -> ImageSubmessage:
    return {"type": "image_url", "image_url": {"url": url}}


def create_text_message(text: str) -> TextSubmessage:
    return {"type": "text", "text": text}


def base64_to_image_url(type: str, base64_image: str) -> str:
    return f"data:{type};base64,{base64_image}"


def get_url_image_type(url: str) -> Optional[str]:
    pattern = r"^data:image\/([^;]+);base64,"
    match = re.match(pattern, url)
    if match is None:
        return None
    return match.group(1)


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

    yield chunk_format(
        build_chunk(id, None, {"role": "assistant"}, created, is_stream)
    )

    async for chunk in stream:
        logger.debug(f"chunk: {json.dumps(chunk)}")

        if "error" in chunk:
            yield chunk
            yield END_CHUNK
            return

        id = chunk.get("id", id)
        usage = chunk.get("usage", usage)
        created = chunk.get("created", created)

        choice: dict = chunk.get("choices", [{}])[0]

        finish_reason = get_finish_reason(choice) or finish_reason
        content = choice.get("delta", {}).get("content")

        if content is None:
            continue

        yield chunk_format(
            build_chunk(id, None, {"content": content}, created, is_stream)
        )

    yield chunk_format(
        build_chunk(id, finish_reason, {}, created, is_stream, usage=usage)
    )
    yield END_CHUNK


def derive_image_content_type(attachment: Any) -> Optional[str]:
    type = attachment.get("type")
    if type is None:
        return None

    if (
        type.startswith("image/")
        and type[len("image/") :] in SUPPORTED_IMAGE_TYPES
    ):
        return type

    if "octet-stream" in type:
        # It's an arbitrary binary file. Trying to guess the type from the URL.
        url = attachment.get("url")
        if url is None:
            return None

        file_ext = url.split(".")[-1]

        if file_ext in SUPPORTED_IMAGE_TYPES:
            return f"image/{file_ext}"

    return None


async def download_image_attachment(
    file_storage: Optional[FileStorage], attachment: Any
) -> Optional[ImageSubmessage]:
    type = derive_image_content_type(attachment)
    if type is None:
        return None

    if "data" in attachment:
        url = base64_to_image_url(type, attachment["data"])
        return create_image_message(url)

    if "url" in attachment:
        attachment_link: str = attachment["url"]

        image_url_type = get_url_image_type(attachment_link)
        if image_url_type is not None:
            if image_url_type in SUPPORTED_IMAGE_TYPES:
                return create_image_message(attachment_link)
            else:
                return None

        try:
            if file_storage is not None:
                url = file_storage.attachment_link_to_url(attachment_link)
                base64_str = await file_storage.download_file_as_base64(url)
            else:
                base64_str = await download_file_as_base64(attachment_link)

            image_url = base64_to_image_url(type, base64_str)
            return create_image_message(image_url)
        except Exception:
            logger.warning("Failed to download image from URL")
            return create_image_message(attachment_link)

    return None


async def transform_message(
    file_storage: Optional[FileStorage], message: dict
) -> dict | str:
    content = message.get("content", "")
    custom_content = message.get("custom_content", {})
    attachments = custom_content.get("attachments", [])

    logger.debug(f"original attachments: {attachments}")

    conversion_results: List[Optional[ImageSubmessage]] = [
        await download_image_attachment(file_storage, attachment)
        for attachment in attachments
    ]

    image_attachments: List[ImageSubmessage] = [
        m for m in conversion_results if m is not None
    ]

    conversion_errors: List[int] = [
        idx for idx, m in enumerate(conversion_results) if m is None
    ]

    logger.debug(f"image attachments: {str(image_attachments)[:100]}")
    logger.debug(f"conversion errors: {conversion_errors}")

    if len(image_attachments) == 0:
        conversion_fails_msg = ""
        if len(conversion_errors) > 0:
            conversion_fails_msg = " None of the provided attachments is a supported image attachment."

        return (
            "No image attachments were found."
            + conversion_fails_msg
            + "\n\n"
            + USAGE
        )

    new_content: VisionContent = [
        create_text_message(content)
    ] + image_attachments

    message = {k: v for k, v in message.items() if k != "custom_content"}
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

    messages: List[Any] = request["messages"]
    if len(messages) == 0:
        raise HTTPException(
            status_code=422,
            message="The request doesn't contain any messages",
            type="invalid_request_error",
        )

    api_url = upstream_endpoint + "?api-version=2023-12-01-preview"

    last_message = messages[-1]
    new_message = await transform_message(file_storage, last_message)

    if isinstance(new_message, str):
        return create_predefined_response(new_message, is_stream)

    max_tokens = request.get("max_tokens", DEFAULT_MAX_TOKENS)

    request = {
        **request,
        "max_tokens": max_tokens,
        "messages": [new_message],
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
        content = response["choices"][0]["message"].get("content", "")
        usage = response["usage"]
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
