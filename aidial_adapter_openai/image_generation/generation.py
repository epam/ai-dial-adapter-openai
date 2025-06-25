from typing import Any, AsyncIterator, Optional, TypeVar

from aidial_sdk.exceptions import InternalServerError, RequestValidationError
from openai.types.image import Image
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import image_gen_parser
from aidial_adapter_openai.utils.streaming import build_chunk, generate_id

IMG_USAGE = {
    "prompt_tokens": 0,
    "completion_tokens": 1,
    "total_tokens": 1,
}


def create_custom_content(image: Image, content_type: str) -> Any:
    attachments = []

    if revised_prompt := image.revised_prompt:
        attachments.append({"title": "Revised prompt", "data": revised_prompt})

    if (data := image.b64_json) is None:
        raise InternalServerError(
            "The model didn't return the base64 encoding of an image"
        )

    attachments.append({"title": "Image", "type": content_type, "data": data})

    return {"custom_content": {"attachments": attachments}}


async def generate_stream(
    id: str, created: int, message_content: Any
) -> AsyncIterator[dict]:
    yield build_chunk(id, None, {"role": "assistant"}, created, True)
    yield build_chunk(id, None, message_content, created, True)
    yield build_chunk(id, "stop", {}, created, True, usage=IMG_USAGE)


def generate_response(id: str, created: int, message_content: Any) -> dict:
    return build_chunk(
        id,
        "stop",
        {"role": "assistant", **message_content},
        created,
        False,
        usage=IMG_USAGE,
    )


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


_Config = TypeVar("_Config", bound=BaseModel)


async def chat_completion(
    model: ImageGenerationModel[_Config],
    data: Any,
    deployment: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    is_stream: bool,
    file_storage: Optional[FileStorage],
    api_version: str,
):
    if data.get("n", 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    client = image_gen_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version}
    )

    user_prompt = get_user_prompt(data)

    config_cls = model.get_configuration()
    config = parse_configuration(config_cls, data) or config_cls()

    model_response = await client.images.generate(
        model=deployment,
        prompt=user_prompt,
        response_format=model.get_response_format(),
        extra_body=config.dict(exclude_none=True),
    )

    if not (data := model_response.data) or len(data) < 1:
        raise InternalServerError("The model didn't return an image")

    image = data[0]

    image_content_type = model.get_image_content_type(config)
    custom_content = create_custom_content(image, image_content_type)
    message_content = {"content": "", **custom_content}

    if file_storage is not None:
        await move_attachments_data_to_storage(custom_content, file_storage)

    id = generate_id()
    created = model_response.created

    if is_stream:
        return generate_stream(id, created, message_content)
    else:
        return generate_response(id, created, message_content)
