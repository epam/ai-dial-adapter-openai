import logging
from typing import Any, assert_never

import fastapi
import openai
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.audio import (
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
)
from openai.types.audio.transcription_create_response import (
    TranscriptionCreateResponse,
)
from openai.types.audio.transcription_stream_event import (
    TranscriptionStreamEvent,
)
from pydantic import BaseModel

from aidial_adapter_openai.audio_api.transcribe.prompt import TranscribePrompt
from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.streaming import generate_created, generate_id


class TokenUsage(BaseModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def set_usage(self, response: DIALResponse):
        response.set_usage(
            prompt_tokens=self.prompt_tokens or 0,
            completion_tokens=self.completion_tokens or 0,
        )


def _get_usage(
    chunk: TranscriptionCreateResponse | TranscriptionTextDoneEvent,
) -> TokenUsage | None:
    # NOTE: whisper has completely different API for its responses
    duration: Any | None = getattr(chunk, "duration", None)
    if duration is not None and isinstance(duration, (float, int)):
        return TokenUsage(prompt_tokens=int(duration))

    usage_dict: dict | None = getattr(chunk, "usage", None)  # type: ignore
    if usage_dict is None:
        return None

    if (type := usage_dict.get("type")) is None:
        return None

    # NOTE: gpt-4o supposed to return usage in tokens, whisper - in seconds,
    # however whisper doesn't return usage field at all.
    match type:
        case "tokens":
            return TokenUsage(
                prompt_tokens=usage_dict.get("input_tokens"),
                completion_tokens=usage_dict.get("output_tokens"),
            )
        case "duration":
            return TokenUsage(prompt_tokens=int(usage_dict.get("seconds") or 0))
        case _:
            logger.error(f"Unknown type of usage: {type!r}.")
            return None


AudioResponse = (
    TranscriptionCreateResponse | openai.AsyncStream[TranscriptionStreamEvent]
)


async def normalize_audio_response(response: AudioResponse) -> AudioResponse:
    """
    Special handling for responses from the Whisper model.
    It ignores stream=true parameter and returns a block JSON response
    which is wrapped by the openai library into AsyncStream.
    """
    if isinstance(response, openai.AsyncStream):
        if "application/json" in response.response.headers["content-type"]:
            response_bytes = await response.response.aread()
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"raw response: {response_bytes!r}")

            return TranscriptionVerbose.parse_raw(response_bytes)

    return response


async def chat_completion(
    *,
    request: fastapi.Request,
    request_body: Any,
    deployment_id: str,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
):
    is_stream = bool(request_body.get("stream"))
    model_name = request_body["model"]

    is_whisper_deployment = "whisper" in model_name
    response_format = "verbose_json" if is_whisper_deployment else "json"

    prompt = await TranscribePrompt.from_request(request_body, file_storage)
    file = (prompt.audio_filename, prompt.audio_data, prompt.audio_type)

    audio_response = await client.audio.transcriptions.create(
        file=file,
        prompt=prompt.system_message or openai.NOT_GIVEN,
        model=model_name,
        stream=is_stream,
        response_format=response_format,
        temperature=request_body.get("temperature") or openai.NOT_GIVEN,
    )

    audio_response = await normalize_audio_response(audio_response)

    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        response.set_model(model_name)
        response.set_response_id(generate_id())
        response.set_created(generate_created())

        with response.create_single_choice() as choice:
            if isinstance(audio_response, openai.AsyncStream):
                async for chunk in audio_response:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"response chunk: {chunk.json()}")

                    match chunk:
                        case TranscriptionTextDeltaEvent(delta=delta):
                            choice.append_content(delta)
                        case TranscriptionTextDoneEvent():
                            if usage := _get_usage(chunk):
                                usage.set_usage(response)
                        case _:
                            assert_never(chunk)
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"response: {audio_response.json()}")

                choice.append_content(audio_response.text)
                if usage := _get_usage(audio_response):
                    usage.set_usage(response)

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
