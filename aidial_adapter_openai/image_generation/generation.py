from typing import Any, TypeVar

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import InternalServerError
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.images_response import ImagesResponse
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.image_generation.prompt import ImageGenPrompt
from aidial_adapter_openai.utils.streaming import generate_id

_Config = TypeVar("_Config", bound=BaseModel)


async def chat_completion(
    *,
    request: fastapi.Request,
    model: ImageGenerationModel[_Config],
    deployment_id: str,
    request_body: Any,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
):
    prompt = await ImageGenPrompt.from_request(request_body, file_storage)

    n = int(request_body.get("n", 1))
    model_name = request_body["model"]

    config_cls = model.get_configuration()
    response_format = model.get_response_format()
    config = parse_configuration(config_cls, request_body) or config_cls()
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

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)
        response.set_response_id(generate_id())
        response.set_created(model_response.created)

        for image in images:
            with response.create_choice() as choice:
                choice.append_content("")

                if prompt := image.revised_prompt:
                    choice.add_attachment(title="Revised prompt", data=prompt)

                if (data_b64 := image.b64_json) is None:
                    raise InternalServerError(
                        "The model didn't return the base64 encoding of an image"
                    )

                if file_storage:
                    metadata = await file_storage.upload_file(
                        "images", data_b64, image_content_type
                    )
                    url = metadata["url"]
                    data = None
                else:
                    url = None
                    data = data_b64

                choice.add_attachment(
                    title="Image", type=image_content_type, url=url, data=data
                )

        response.set_usage(prompt_tokens=0, completion_tokens=n)

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
