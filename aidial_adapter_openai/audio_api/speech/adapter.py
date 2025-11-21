import base64
from typing import Any

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI

from aidial_adapter_openai.audio_api.speech.configuration import Configuration
from aidial_adapter_openai.dial_api.attachment import create_dial_attachment
from aidial_adapter_openai.dial_api.request import (
    collect_message_text_content,
    parse_configuration,
)
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.streaming import generate_created, generate_id
from aidial_adapter_openai.utils.tokenizer import Tokenizer


def create_assistant_message(data: bytes, content_type: str) -> dict:
    data_base64 = base64.b64encode(data).decode()
    return {
        "role": "assistant",
        "content": "",
        "custom_content": {
            "attachments": [
                {"title": "Audio", "type": content_type, "data": data_base64}
            ]
        },
    }


def collect_system_messages(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


async def chat_completion(
    *,
    request: fastapi.Request,
    request_data: Any,
    deployment_id: str,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
):
    if (n := request_data.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    messages = request_data.pop("messages")
    if not messages:
        raise RequestValidationError("The request doesn't contain any messages")

    prompt = collect_message_text_content(messages[-1]).strip()
    prompt_tokens = await tokenizer.tokenize_text(prompt)

    model_name = request_data["model"]

    config = parse_configuration(Configuration, request_data) or Configuration()

    if system_message := collect_system_messages(messages):
        config.instructions = (
            system_message + "\n" + (config.instructions or "")
        ).strip() or None

    extra_body = config.dict(exclude_none=True)

    response = await client.audio.speech.create(
        input=prompt, model=model_name, **extra_body
    )

    audio_data = response.read()
    audio_format = response.response.headers.get("content-type") or "audio/mpeg"

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)
        response.set_response_id(generate_id())
        response.set_created(generate_created())

        with response.create_single_choice() as choice:
            data_b64 = base64.b64encode(audio_data).decode()

            choice.append_content("")
            choice.add_attachment(
                await create_dial_attachment(
                    title="Audio",
                    content_type=audio_format,
                    data=data_b64,
                    file_storage=file_storage,
                    upload_dir="audio",
                )
            )

        response.set_usage(prompt_tokens=prompt_tokens, completion_tokens=0)

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
