from typing import Any, Dict

from aidial_sdk.exceptions import RequestValidationError
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai.types import Completion

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    debug_print,
    map_stream,
)


def sanitize_text(text: str) -> str:
    return text.replace("<|endoftext|>", "")


def convert_to_chat_completions_response(
    chunk: Completion, is_stream: bool
) -> Dict[str, Any]:
    converted_chunk = build_chunk(
        id=chunk.id,
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
    request: Dict[str, Any],
    client: AsyncAzureOpenAI | AsyncOpenAI,
    deployment_id: str,
    app_config: ApplicationConfig,
):
    if (request.get("n") or 1) > 1:
        raise RequestValidationError("The deployment doesn't support n > 1")

    messages = request.get("messages") or []
    if not messages:
        raise RequestValidationError("The request doesn't contain any messages")

    prompt = messages[-1].get("content") or ""

    if (
        template := app_config.COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES.get(
            deployment_id
        )
    ) is not None:
        prompt = template.format(prompt=prompt)

    del request["messages"]

    response = await call_with_extra_body(
        client.completions.create,
        {"prompt": prompt, **request},
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
