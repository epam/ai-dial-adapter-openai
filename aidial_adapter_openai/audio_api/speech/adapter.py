import base64
from typing import Any, AsyncIterator

from aidial_sdk.exceptions import RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI

from aidial_adapter_openai.audio_api.speech.configuration import Configuration
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


def collect_system_messages(messages: list[dict]) -> str | None:
    ret = ""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            ret += collect_message_text_content(message)
    return ret.strip() or None


async def chat_completion(
    *,
    request: Any,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    file_storage: FileStorage | None,
    tokenizer: Tokenizer,
):
    if (n := request.get("n")) not in [None, 1]:
        raise RequestValidationError(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    messages = request.pop("messages")
    if not messages:
        raise RequestValidationError("The request doesn't contain any messages")

    prompt = collect_message_text_content(messages[-1]).strip()
    prompt_tokens = await tokenizer.tokenize_text(prompt)

    is_stream = bool(request.get("stream"))
    model_name = request["model"]

    config = parse_configuration(Configuration, request) or Configuration()

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

    message = create_assistant_message(audio_data, audio_format)
    await upload_message_attachments_to_storage(file_storage, "audio", message)

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
