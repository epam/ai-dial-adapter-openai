from typing import Any, AsyncIterator, TypeVar

from aidial_sdk.exceptions import InternalServerError, RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.image import Image
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.image_generation.prompt import ImageGenPrompt
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


async def upload_attachments_data_to_storage(
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
    client: AsyncAzureOpenAI | AsyncOpenAI,
    is_stream: bool,
    file_storage: FileStorage | None,
):
    if data.get("n", 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    prompt = await ImageGenPrompt.from_request(data, file_storage)

    config_cls = model.get_configuration()
    response_format = model.get_response_format()
    config = parse_configuration(config_cls, data) or config_cls()
    extra_body = config.dict(exclude_none=True)

    images = [
        (f"image_{i}", resource.data, resource.type)
        for (i, resource) in enumerate(prompt.images)
    ]

    if prompt.images:
        model_response = await client.images.edit(
            model=deployment,
            image=images,  # type: ignore
            prompt=prompt.text_prompt,
            response_format=response_format,
            extra_body=extra_body,
        )
    else:
        model_response = await client.images.generate(
            model=deployment,
            prompt=prompt.text_prompt,
            response_format=response_format,
            extra_body=extra_body,
        )

    images = model_response.data

    if not images:
        raise InternalServerError("The model didn't return an image")

    image = images[0]
    image_content_type = model.get_image_content_type(config)
    custom_content = create_custom_content(image, image_content_type)
    message_content = {"content": "", **custom_content}

    if file_storage is not None:
        await upload_attachments_data_to_storage(custom_content, file_storage)

    id = generate_id()
    created = model_response.created

    if is_stream:
        return generate_stream(id, created, message_content)
    else:
        return generate_response(id, created, message_content)
