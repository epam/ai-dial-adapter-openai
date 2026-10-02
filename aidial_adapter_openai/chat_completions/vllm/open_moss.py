import base64
from typing import Any

import fastapi
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from aidial_sdk.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI
from pydantic import Field

from aidial_adapter_openai.audio_api.speech.adapter import (
    collect_system_messages,
)
from aidial_adapter_openai.audio_api.speech.configuration import (
    Configuration as SpeechConfiguration,
)
from aidial_adapter_openai.dial_api.attachment import create_dial_attachment
from aidial_adapter_openai.dial_api.request import (
    collect_message_text_content,
    parse_configuration,
)
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.streaming import generate_created, generate_id
from aidial_adapter_openai.utils.tokenizer import Tokenizer


class OpenMossConfiguration(SpeechConfiguration):
    ref_audio: str | None = Field(
        default=None,
        description=(
            "Reference audio for voice cloning. "
            "Accepts a data URL (data:audio/wav;base64,...) or a public URL."
        ),
    )
    ref_audio_2: str | None = Field(
        default=None,
        description=(
            "Secondary reference audio for voice cloning. "
            "Accepts a data URL (data:audio/wav;base64,...) or a public URL."
        ),
    )
    ref_audio_3: str | None = Field(
        default=None,
        description=(
            "Third reference audio for voice cloning. "
            "Accepts a data URL (data:audio/wav;base64,...) or a public URL."
        ),
    )
    ref_audio_4: str | None = Field(
        default=None,
        description=(
            "Fourth reference audio for voice cloning. "
            "Accepts a data URL (data:audio/wav;base64,...) or a public URL."
        ),
    )
    ref_audio_5: str | None = Field(
        default=None,
        description=(
            "Fifth reference audio for voice cloning. "
            "Accepts a data URL (data:audio/wav;base64,...) or a public URL."
        ),
    )
    ref_text: str | None = Field(
        default=None,
        description="Reference text corresponding to the reference audio.",
    )
    max_new_tokens: int | None = Field(
        default=None,
        description="Maximum number of new tokens to generate.",
    )
    seed: int | None = Field(
        default=None,
        description="Random seed for reproducible generation.",
    )


_STANDARD_SPEECH_FIELDS = {
    "voice",
    "speed",
    "response_format",
    "instructions",
}


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Any,
    deployment_id: str,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
) -> StreamingResponse | dict:
    if (n := request_body.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    if not (messages := request_body.get("messages")):
        raise RequestValidationError("The request doesn't contain any messages")

    prompt = collect_message_text_content(messages[-1]).strip()
    prompt_tokens = await tokenizer.tokenize_text(prompt)

    config = (
        parse_configuration(OpenMossConfiguration, request_body)
        or OpenMossConfiguration()
    )

    if system_message := collect_system_messages(messages):
        config.instructions = (
            system_message + "\n" + (config.instructions or "")
        ).strip() or None

    fields = config.model_dump(exclude_none=True)
    standard_kwargs = {
        key: value
        for key, value in fields.items()
        if key in _STANDARD_SPEECH_FIELDS
    }
    extra_body = {
        key: value
        for key, value in fields.items()
        if key not in _STANDARD_SPEECH_FIELDS
    }

    response = await client.audio.speech.create(
        input=prompt,
        model=deployment_id,
        extra_body=extra_body or None,
        **standard_kwargs,
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
