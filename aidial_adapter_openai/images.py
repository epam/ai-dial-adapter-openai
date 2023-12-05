from typing import Any, AsyncGenerator, Optional

import aiohttp
from fastapi import Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.storage import FileStorage, upload_base64_file
from aidial_adapter_openai.utils.streaming import (
    END_CHUNK,
    build_chunk,
    chunk_format,
    generate_id,
)


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


def build_custom_content(base64_image: str, revised_prompt: str) -> Any:
    return {
        "custom_content": {
            "attachments": [
                {"title": "Image", "type": "image/png", "data": base64_image},
                {"title": "Revised prompt", "data": revised_prompt},
            ]
        }
    }


async def generate_stream(
    id: str, created: str, custom_content: Any
) -> AsyncGenerator[Any, Any]:
    yield chunk_format(
        build_chunk(id, None, {"role": "assistant"}, created, True)
    )

    yield chunk_format(build_chunk(id, None, custom_content, created, True))

    yield chunk_format(
        build_chunk(
            id,
            "stop",
            {},
            created,
            True,
            usage={
                "completion_tokens": 0,
                "prompt_tokens": 1,
                "total_tokens": 1,
            },
        )
    )

    yield END_CHUNK


def validate_request_data(data: Any):
    if (
        "messages" not in data
        or len(data["messages"]) == 0
        or "content" not in data["messages"][-1]
    ):
        raise HTTPException(
            "Your request is invalid", 400, "invalid_request_error"
        )


async def text_to_image_chat_completion(
    data: Any,
    upstream_endpoint: str,
    api_key: str,
    is_stream: bool,
    file_storage: Optional[FileStorage],
) -> Response:
    validate_request_data(data)

    api_url = upstream_endpoint + "?api-version=2023-12-01-preview"
    user_prompt = data["messages"][-1]["content"]
    model_response = await generate_image(api_url, api_key, user_prompt)

    if isinstance(model_response, JSONResponse):
        return model_response

    base64_image = model_response["data"][0]["b64_json"]
    revised_prompt = model_response["data"][0]["revised_prompt"]

    id = generate_id()
    created = model_response["created"]

    custom_content = build_custom_content(base64_image, revised_prompt)

    if file_storage is not None:
        file_metadata = await upload_base64_file(
            file_storage, base64_image, "image/png"
        )
        image_url = file_metadata["path"] + "/" + file_metadata["name"]

        del custom_content["custom_content"]["attachments"][0]["data"]
        custom_content["custom_content"]["attachments"][0]["url"] = image_url

    if not is_stream:
        return JSONResponse(
            content=build_chunk(
                id,
                "stop",
                {**custom_content, "role": "assistant"},
                created,
                False,
                usage={
                    "completion_tokens": 0,
                    "prompt_tokens": 1,
                    "total_tokens": 1,
                },
            )
        )
    else:
        return StreamingResponse(
            generate_stream(id, created, custom_content),
            media_type="text/event-stream",
        )
