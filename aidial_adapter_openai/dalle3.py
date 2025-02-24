from typing import Any, AsyncIterator, Literal, Optional

import aiohttp
from aidial_sdk.exceptions import HTTPException as DIALException
from aidial_sdk.exceptions import RequestValidationError
from aidial_sdk.pydantic_v1 import BaseModel, Field
from fastapi.responses import JSONResponse

from aidial_adapter_openai.dial_api.request import get_configuration
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds, get_auth_headers
from aidial_adapter_openai.utils.streaming import build_chunk, generate_id

IMG_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 1,
    "total_tokens": 1,
}


class Dalle3Config(BaseModel):
    quality: Optional[Literal["standard", "hd"] | str] = Field(
        default=None,
        description="The quality of the image that will be generated.",
    )

    size: Optional[Literal["1024x1024", "1792x1024", "1024x1792"] | str] = (
        Field(default=None, description="The size of the generated images.")
    )

    style: Optional[Literal["vivid", "natural"] | str] = Field(
        default=None, description="The style of the generated images."
    )


async def generate_image(
    api_url: str, creds: OpenAICreds, config: Dalle3Config, user_prompt: str
) -> JSONResponse | Any:
    async with aiohttp.ClientSession() as session:
        async with session.post(
            api_url,
            json={
                "prompt": user_prompt,
                "response_format": "b64_json",
                **config.dict(exclude_none=True),
            },
            headers=get_auth_headers(creds),
        ) as response:
            status_code = response.status

            data = await response.json()

            if status_code == 200:
                return data

            if "error" in data:
                error = data["error"]

                if error.get("code") in [
                    "content_policy_violation",
                    "contentFilter",
                ]:
                    error["code"] = "content_filter"

                return DIALException(
                    status_code=status_code,
                    message=error.get("message"),
                    type=error.get("type"),
                    param=error.get("param"),
                    code=error.get("code"),
                ).to_fastapi_response()
            else:
                return JSONResponse(content=data, status_code=status_code)


def build_custom_content(base64_image: str, revised_prompt: str) -> Any:
    return {
        "custom_content": {
            "attachments": [
                {"title": "Revised prompt", "data": revised_prompt},
                {"title": "Image", "type": "image/png", "data": base64_image},
            ]
        },
        "content": "",
    }


async def generate_stream(
    id: str, created: int, custom_content: Any
) -> AsyncIterator[dict]:
    yield build_chunk(id, None, {"role": "assistant"}, created, True)
    yield build_chunk(id, None, custom_content, created, True)
    yield build_chunk(id, "stop", {}, created, True, usage=IMG_USAGE)


def get_user_prompt(data: Any) -> str:
    try:
        prompt = data["messages"][-1]["content"]
        if not isinstance(prompt, str):
            raise ValueError("Content isn't a string")
        return prompt
    except Exception as e:
        raise RequestValidationError(
            "Invalid request. Expected a string at path 'messages[-1].content'."
        ) from e


async def move_attachments_data_to_storage(
    custom_content: Any, file_storage: FileStorage
):
    for attachment in custom_content["custom_content"]["attachments"]:
        if (
            "data" not in attachment
            or "type" not in attachment
            or not attachment["type"].startswith("image/")
        ):
            continue

        file_metadata = await file_storage.upload_file_as_base64(
            attachment["data"], attachment["type"]
        )

        del attachment["data"]
        attachment["url"] = file_metadata["url"]


async def chat_completion(
    data: Any,
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
):
    if data.get("n", 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    api_url = f"{upstream_endpoint}?api-version={api_version}"
    user_prompt = get_user_prompt(data)

    config = get_configuration(Dalle3Config, data) or Dalle3Config()
    model_response = await generate_image(api_url, creds, config, user_prompt)

    if isinstance(model_response, JSONResponse):
        return model_response

    base64_image = model_response["data"][0]["b64_json"]
    revised_prompt = model_response["data"][0]["revised_prompt"]

    id = generate_id()
    created = model_response["created"]

    custom_content = build_custom_content(base64_image, revised_prompt)

    if file_storage is not None:
        await move_attachments_data_to_storage(custom_content, file_storage)

    if is_stream:
        return generate_stream(id, created, custom_content)
    else:
        return build_chunk(
            id,
            "stop",
            {"role": "assistant", "content": "", **custom_content},
            created,
            False,
            usage=IMG_USAGE,
        )
