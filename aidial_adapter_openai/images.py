from typing import Any, AsyncGenerator, Optional

import aiohttp
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.storage import FileStorage, upload_base64_file
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
    generate_id,
)

IMG_USAGE = {
    "completion_tokens": 0,
    "prompt_tokens": 1,
    "total_tokens": 1,
}


async def generate_image(api_url: str, api_key: str, user_prompt: str) -> Any:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            json={"prompt": user_prompt, "response_format": "b64_json"},
            headers={"api-key": api_key},
        ) as response:
            status_code = response.status

            data = await response.json()

            if status_code == 200:
                return data
            else:
                return JSONResponse(content=data, status_code=status_code)


def build_common_custom_content(revised_prompt: str):
    return {
        "custom_content": {
            "attachments": [{"title": "Revised prompt", "data": revised_prompt}]
        }
    }


def build_custom_content_with_base64(
    base64_image: str, revised_prompt: str
) -> Any:
    custom_content = build_common_custom_content(revised_prompt)

    custom_content["custom_content"]["attachments"].append(
        {"title": "Image", "type": "image/png", "data": base64_image}
    )

    return custom_content


def build_custom_content_with_url(url_image: str, revised_prompt: str) -> Any:
    custom_content = build_common_custom_content(revised_prompt)

    custom_content["custom_content"]["attachments"].append(
        {"title": "Image", "type": "image/png", "url": url_image},
    )

    return custom_content


async def generate_stream(
    id: str, created: str, custom_content: Any
) -> AsyncGenerator[Any, Any]:
    yield chunk_format(
        build_chunk(id, None, {"role": "assistant"}, created, True)
    )

    yield chunk_format(build_chunk(id, None, custom_content, created, True))

    yield chunk_format(
        build_chunk(id, "stop", {}, created, True, usage=IMG_USAGE)
    )

    yield END_CHUNK


def get_user_prompt(data: Any):
    if (
        "messages" not in data
        or len(data["messages"]) == 0
        or "content" not in data["messages"][-1]
        or not data["messages"][-1]
    ):
        raise HTTPException(
            "Your request is invalid", 400, "invalid_request_error"
        )

    return data["messages"][-1]["content"]


async def text_to_image_chat_completion(
    data: Any,
    upstream_endpoint: str,
    api_key: str,
    is_stream: bool,
    file_storage: Optional[FileStorage],
) -> Response:
    if data.get("n", 1) > 1:
        raise HTTPException(
            status_code=422,
            message="The deployment doesn't support n > 1",
            type="invalid_request_error",
        )

    api_url = upstream_endpoint + "?api-version=2023-12-01-preview"
    user_prompt = get_user_prompt(data)
    model_response = await generate_image(api_url, api_key, user_prompt)

    if isinstance(model_response, JSONResponse):
        return model_response

    base64_image = model_response["data"][0]["b64_json"]
    revised_prompt = model_response["data"][0]["revised_prompt"]

    id = generate_id()
    created = model_response["created"]

    if file_storage is not None:
        file_metadata = await upload_base64_file(
            file_storage, base64_image, "image/png"
        )
        image_url = file_metadata["path"] + "/" + file_metadata["name"]

        custom_content = build_custom_content_with_url(
            image_url, revised_prompt
        )
    else:
        custom_content = build_custom_content_with_base64(
            base64_image, revised_prompt
        )

    if not is_stream:
        return JSONResponse(
            content=build_chunk(
                id,
                "stop",
                {**custom_content, "role": "assistant"},
                created,
                False,
                usage=IMG_USAGE,
            )
        )
    else:
        return StreamingResponse(
            generate_stream(id, created, custom_content),
            media_type="text/event-stream",
        )
