import base64
from typing import Any, AsyncIterator, assert_never

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
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    generate_created,
    generate_id,
)


def _create_usage(prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def collect_system_messages(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


async def download_audio_file(
    file_storage: FileStorage | None, message: dict
) -> bytes:
    if cc := message.get("custom_content"):
        if attachments := cc.get("attachments"):
            for attachment in attachments:
                if attachment.get("type", "").startswith("audio/"):
                    if data_base64 := attachment.get("data"):
                        return base64.b64decode(data_base64)
                    elif url := attachment.get("url"):
                        if file_storage is not None:
                            return await file_storage.download_file(url)
                        else:
                            return await download_file(url)

    raise RequestValidationError(
        "No audio attachment found in the last message"
    )


def _extract_usage(
    chunk: TranscriptionTextDoneEvent | Transcription | TranscriptionVerbose,
) -> dict | None:
    usage_dict: dict | None = getattr(chunk, "usage", None)  # type: ignore
    if usage_dict is None:
        return None

    return _create_usage(
        prompt_tokens=usage_dict.get("input_tokens") or 0,
        completion_tokens=usage_dict.get("output_tokens") or 0,
    )


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

    system_message = collect_system_messages(messages)
    audio_file = await download_audio_file(file_storage, messages[-1])

    response = await client.audio.transcriptions.create(
        file=audio_file,
        prompt=system_message or openai.NOT_GIVEN,
        model=model_name,
        stream=is_stream,
        temperature=request.get("temperature") or openai.NOT_GIVEN,
    )

    id, created = generate_id(), generate_created()

    def gen_chunk(
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
            yield gen_chunk(role="assistant")

            async for chunk in response:
                match chunk:
                    case TranscriptionTextDeltaEvent(delta=delta):
                        yield gen_chunk(content=delta)

                    case TranscriptionTextDoneEvent():
                        yield gen_chunk(
                            content="",
                            usage=_extract_usage(chunk),
                            finish_reason="stop",
                        )

                    case _:
                        assert_never(chunk)

        return _gen()
    else:
        return gen_chunk(
            role="assistant",
            content=response.text,
            usage=_extract_usage(response),
            finish_reason="stop",
        )
