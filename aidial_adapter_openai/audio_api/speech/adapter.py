import base64

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI

from aidial_adapter_openai.audio_api.speech.configuration import Configuration
from aidial_adapter_openai.dial_api.attachment import create_dial_attachment
from aidial_adapter_openai.dial_api.request import (
    collect_message_text_content,
    collect_system_messages,
    parse_configuration,
)
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.streaming import generate_created, generate_id
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.validation import RequestValue


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: RequestValue,
    deployment_id: str,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
) -> StreamingResponse | dict:
    if (n := request_body.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    messages = request_body["messages"]
    prompt = collect_message_text_content(messages[-1]).strip()
    prompt_tokens = await tokenizer.tokenize_text(prompt)

    config = parse_configuration(Configuration, request_body) or Configuration()

    if system_message := collect_system_messages(messages):
        config.instructions = (
            system_message + "\n" + (config.instructions or "")
        ).strip() or None

    response = await client.audio.speech.create(
        input=prompt,
        model=deployment_id,
        extra_body=config.model_extra or {},
        **config.model_dump(
            exclude_none=True, include=set(Configuration.model_fields)
        ),
    )

    audio_data = response.read()
    audio_format = response.response.headers.get("content-type") or "audio/mpeg"

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(deployment_id)
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
