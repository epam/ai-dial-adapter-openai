import base64
import logging
import mimetypes
from typing import Any, AsyncIterator, Tuple, assert_never

import openai
from aidial_sdk.exceptions import RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI
from openai.types.audio import (
    Transcription,
    TranscriptionTextDeltaEvent,
    TranscriptionTextDoneEvent,
    TranscriptionVerbose,
)

from aidial_adapter_openai.dial_api.request import collect_message_text_content
from aidial_adapter_openai.dial_api.storage import FileStorage, download_file
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    generate_created,
    generate_id,
)


def collect_system_messages(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


async def download_audio_file(
    file_storage: FileStorage | None, message: dict
) -> Tuple[Resource, str]:

    async def _download_file(attachment: dict) -> bytes | None:
        if data_base64 := attachment.get("data"):
            return base64.b64decode(data_base64)
        elif url := attachment.get("url"):
            if file_storage is not None:
                return await file_storage.download_file(url)
            else:
                return await download_file(url)
        else:
            return None

    if cc := message.get("custom_content"):
        if attachments := cc.get("attachments"):
            for attachment in attachments:
                type = attachment.get("type") or ""
                if not type.startswith("audio/"):
                    continue

                if (data := await _download_file(attachment)) is None:
                    continue

                ext = mimetypes.guess_extension(type) or "audio/mp3"

                return Resource(type=type, data=data), f"file.{ext}"

    raise RequestValidationError(
        "No audio attachment found in the last message"
    )


def _create_usage(
    *,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
) -> dict:
    prompt = prompt_tokens or 0
    completion = completion_tokens or 0
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }


def _get_usage(
    chunk: TranscriptionTextDoneEvent | Transcription | TranscriptionVerbose,
) -> dict | None:
    # NOTE: whisper has completely different API of responses
    duration: Any | None = getattr(chunk, "duration", None)
    if duration is not None and isinstance(duration, (float, int)):
        return _create_usage(prompt_tokens=int(duration))

    usage_dict: dict | None = getattr(chunk, "usage", None)  # type: ignore
    if usage_dict is None:
        return None

    if (type := usage_dict.get("type")) is None:
        return None

    # NOTE: gpt-4o returns usage in tokens, whisper - in seconds.
    if type == "tokens":
        return _create_usage(
            prompt_tokens=usage_dict.get("input_tokens"),
            completion_tokens=usage_dict.get("output_tokens"),
        )

    elif type == "duration":
        return _create_usage(prompt_tokens=int(usage_dict.get("seconds") or 0))

    else:
        logger.error(f"Unknown type of usage: {type!r}.")
        return None


async def chat_completion(
    *,
    request: Any,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
):
    if (n := request.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    messages = request.pop("messages")
    if not messages:
        raise RequestValidationError("The request doesn't contain any messages")

    is_stream = bool(request.get("stream"))
    model_name = request["model"]

    is_whisper_deployment = "whisper" in model_name
    response_format = "verbose_json" if is_whisper_deployment else "json"

    system_message = collect_system_messages(messages)
    resource, filename = await download_audio_file(file_storage, messages[-1])

    response = await client.audio.transcriptions.create(
        file=(filename, resource.data, resource.type),
        prompt=system_message or openai.NOT_GIVEN,
        model=model_name,
        stream=is_stream,
        response_format=response_format,
        temperature=request.get("temperature") or openai.NOT_GIVEN,
    )

    id, created = generate_id(), generate_created()

    def create_chunk(
        *,
        finish_reason: str | None = None,
        role: str | None = None,
        content: str | None = None,
        usage: dict | None = None,
    ) -> dict:
        message = {}
        if role is not None:
            message["role"] = role
        if content is not None:
            message["content"] = content

        return build_chunk(
            id=id,
            created=created,
            model=model_name,
            finish_reason=finish_reason,
            message=message,
            is_stream=is_stream,
            usage=usage,
        )

    if isinstance(response, openai.AsyncStream):

        async def _gen() -> AsyncIterator[dict]:
            yield create_chunk(role="assistant")

            async for chunk in response:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"response chunk: {chunk.json()}")

                match chunk:
                    case TranscriptionTextDeltaEvent(delta=delta):
                        yield create_chunk(content=delta)

                    case TranscriptionTextDoneEvent():
                        yield create_chunk(
                            content="",
                            usage=_get_usage(chunk),
                            finish_reason="stop",
                        )

                    case _:
                        assert_never(chunk)

        return _gen()
    else:
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"response: {response.json()}")

        return create_chunk(
            role="assistant",
            content=response.text,
            usage=_get_usage(response),
            finish_reason="stop",
        )
