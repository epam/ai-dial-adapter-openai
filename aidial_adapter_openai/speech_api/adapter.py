import base64
from typing import Any, AsyncIterator, Literal

from aidial_sdk.exceptions import RequestValidationError
from openai import NOT_GIVEN, AsyncAzureOpenAI, AsyncOpenAI
from pydantic import BaseModel, Field

from aidial_adapter_openai.dial_api.attachment import (
    upload_message_attachments_to_storage,
)
from aidial_adapter_openai.dial_api.request import (
    collect_message_text_content,
    parse_configuration,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    generate_created,
    generate_id,
)
from aidial_adapter_openai.utils.tokenizer import Tokenizer


def _get_usage(prompt_tokens: int) -> dict:
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": 0,
        "total_tokens": prompt_tokens,
    }


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


def collect_instructions(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


Voices = Literal[
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "fable",
    "onyx",
    "nova",
    "sage",
    "shimmer",
    "verse",
]

Formats = Literal["mp3", "opus", "aac", "flac", "wav", "pcm"]


class Configuration(BaseModel):
    instructions: str | None = Field(
        default=None,
        description=(
            "Control the voice of your generated audio with additional instructions. "
            "Does not work with `tts-1` or `tts-1-hd`. "
            "The instruction from the system and developer messages "
            "will be attached to the instructions from the configuration."
        ),
    )
    voice: str | Voices | None = Field(
        default="alloy",
        description="The voice to use when generating the audio.",
    )
    speed: float | None = Field(
        default=None,
        description=(
            "The speed of the generated audio. "
            "Select a value from `0.25` to `4.0`. `1.0` is the default. "
            "Does not work with `gpt-4o-mini-tts`."
        ),
    )
    response_format: str | Formats | None = Field(
        default=None, description="The format of the generated audio."
    )


async def chat_completion(
    *,
    request: Any,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
):
    n = int(request.get("n", 1))
    if n != 1:
        raise RequestValidationError(
            "The deployment doesn't support n other than 1."
        )

    messages = request.pop("messages")
    if not messages:
        raise RequestValidationError("The request doesn't contain any messages")

    prompt = collect_message_text_content(messages[-1]).strip()
    prompt_tokens = await tokenizer.tokenize_text(prompt)

    is_stream = bool(request.get("stream"))
    model_name = request["model"]

    config = parse_configuration(Configuration, request) or Configuration()

    instructions = collect_instructions(messages) or ""
    instructions += config.instructions or ""
    instructions = instructions.strip()

    extra_body = config.dict(exclude_none=True)

    response = await client.audio.speech.create(
        input=prompt,
        model=model_name,
        instructions=instructions or NOT_GIVEN,
        **extra_body,
    )

    audio_data = response.read()
    audio_format = response.response.headers.get("content-type") or "audio/mpeg"

    message = create_assistant_message(audio_data, audio_format)
    await upload_message_attachments_to_storage(file_storage, message)

    chunk = build_chunk(
        id=generate_id(),
        created=generate_created(),
        model=model_name,
        finish_reason="stop",
        message=message,
        is_stream=is_stream,
        usage=_get_usage(prompt_tokens),
    )

    if is_stream:

        async def _gen() -> AsyncIterator[dict]:
            yield chunk

        return _gen()
    else:
        return chunk
