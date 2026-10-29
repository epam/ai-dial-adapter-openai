from collections.abc import AsyncIterator
from typing import Any

from aidial_sdk.exceptions import RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types import Completion

from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    debug_print,
    map_stream,
)
from aidial_adapter_openai.utils.validation import RequestValue


def sanitize_text(text: str) -> str:
    return text.replace("<|endoftext|>", "")


def convert_to_chat_completions_response(
    chunk: Completion, is_stream: bool
) -> dict[str, Any]:
    converted_chunk = build_chunk(
        id=chunk.id,
        model=chunk.model,
        finish_reason=chunk.choices[0].finish_reason,
        message={
            "content": sanitize_text(chunk.choices[0].text),
            "role": "assistant",
        },
        created=chunk.created,
        is_stream=is_stream,
        usage=chunk.usage.to_dict() if chunk.usage else None,
    )
    debug_print("response", converted_chunk)
    return converted_chunk


async def chat_completion(
    *,
    request: RequestValue,
    client: AsyncAzureOpenAI | AsyncOpenAI,
    prompt_template: str | None,
) -> AsyncIterator[dict] | dict:
    if (request.get_int("n") or 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    if not (messages := request["messages"].to_list()):
        raise RequestValidationError("The request doesn't contain any messages")

    content = messages[-1].get("content")
    prompt = content.unwrap() if content else ""

    if prompt_template is not None:
        prompt = prompt_template.format(prompt=prompt)

    raw_request = request.unwrap_dict()
    # FIX_NOW: add API for pop
    raw_request.pop("messages")
    response = await call_with_extra_body(
        client.completions.create,
        {"prompt": prompt, **raw_request},
    )

    if isinstance(response, AsyncStream):
        return map_stream(
            lambda item: convert_to_chat_completions_response(
                item, is_stream=True
            ),
            response,
        )
    else:
        return convert_to_chat_completions_response(response, is_stream=False)
