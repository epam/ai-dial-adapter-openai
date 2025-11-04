from typing import Any, AsyncIterator, List, TypeVar

from aidial_sdk.exceptions import InternalServerError
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.image import Image
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.attachment import (
    upload_message_attachments_to_storage,
)
from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.image_generation.prompt import ImageGenPrompt
from aidial_adapter_openai.utils.streaming import build_chunk, generate_id


def _get_usage(n: int) -> dict:
    return {
        "prompt_tokens": 0,
        "completion_tokens": n,
        "total_tokens": n,
    }


def create_custom_content(image: Image, content_type: str) -> dict:
    attachments = []

    if revised_prompt := image.revised_prompt:
        attachments.append({"title": "Revised prompt", "data": revised_prompt})

    if (data := image.b64_json) is None:
        raise InternalServerError(
            "The model didn't return the base64 encoding of an image"
        )

    attachments.append({"title": "Image", "type": content_type, "data": data})

    return {"attachments": attachments}


def create_assistant_messages(images: List[Image], content_type: str):
    for image in images:
        custom_content = create_custom_content(image, content_type)
        yield {
            "role": "assistant",
            "content": "",
            "custom_content": custom_content,
        }


_Config = TypeVar("_Config", bound=BaseModel)


async def chat_completion(
    *,
    model: ImageGenerationModel[_Config],
    request: Any,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
):

    prompt = await ImageGenPrompt.from_request(request, file_storage)

    n = int(request.get("n", 1))
    is_stream = bool(request.get("stream"))
    model_name = request["model"]

    config_cls = model.get_configuration()
    response_format = model.get_response_format()
    config = parse_configuration(config_cls, request) or config_cls()
    extra_body = config.dict(exclude_none=True)

    images = [
        (f"image_{i}", resource.data, resource.type)
        for (i, resource) in enumerate(prompt.images)
    ]

    if prompt.images:
        model_response: ImagesResponse = await client.images.edit(
            model=model_name,
            image=images,  # type: ignore
            prompt=prompt.text_prompt,
            response_format=response_format,
            n=n,
            extra_body=extra_body,
        )
    else:
        model_response = await client.images.generate(
            model=model_name,
            prompt=prompt.text_prompt,
            response_format=response_format,
            n=n,
            extra_body=extra_body,
        )

    if not (images := model_response.data):
        raise InternalServerError("The model didn't return an image")

    image_content_type = model.get_image_content_type(config)
    messages = list(create_assistant_messages(images, image_content_type))

    for message in messages:
        await upload_message_attachments_to_storage(
            file_storage, "images", message
        )

    id = generate_id()
    created = model_response.created

    chunk = build_chunk(
        id=id,
        model=model_name,
        finish_reason="stop",
        message=messages,
        created=created,
        is_stream=is_stream,
        usage=_get_usage(n),
    )

    if is_stream:

        async def _gen() -> AsyncIterator[dict]:
            yield chunk

        return _gen()
    else:
        return chunk
