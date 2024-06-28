from typing import Any, Dict

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi.responses import JSONResponse, StreamingResponse
from openai import AsyncStream
from openai.types import Completion

from aidial_adapter_openai.env import COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import (
    build_chunk,
    debug_print,
    map_stream,
)


def convert_to_chat_completions_response(
    chunk: Completion, is_stream: bool
) -> Dict[str, Any]:
    converted_chunk = build_chunk(
        id=chunk.id,
        finish_reason=chunk.choices[0].finish_reason,
        delta={"content": chunk.choices[0].text, "role": "assistant"},
        created=str(chunk.created),
        is_stream=is_stream,
        usage=chunk.usage.to_dict() if chunk.usage else None,
    )
    debug_print("response", converted_chunk)
    return converted_chunk


async def chat_completion(
    data: Dict[str, Any],
    endpoint: OpenAIEndpoint | AzureOpenAIEndpoint,
    creds: OpenAICreds,
    api_version: str,
    deployment_id: str,
) -> Any:

    if data.get("n", 1) > 1:  # type: ignore
        raise DialException(
            status_code=422,
            message="The deployment doesn't support n > 1",
            type="invalid_request_error",
        )

    client = endpoint.get_client({**creds, "api_version": api_version})

    messages = data.get("messages", [])
    if len(messages) == 0:
        raise DialException(
            status_code=422,
            message="The request doesn't contain any messages",
            type="invalid_request_error",
        )

    prompt = messages[-1].get("content") or ""

    if (
        template := COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES.get(deployment_id)
    ) is not None:
        prompt = template.format(prompt=prompt)

    del data["messages"]
    response = await call_with_extra_body(
        client.completions.create,
        {"prompt": prompt, **data},
    )

    if isinstance(response, AsyncStream):
        return StreamingResponse(
            to_openai_sse_stream(
                map_stream(
                    lambda item: convert_to_chat_completions_response(
                        item, is_stream=True
                    ),
                    response,
                )
            ),
            media_type="text/event-stream",
        )
    else:
        return JSONResponse(
            convert_to_chat_completions_response(response, is_stream=False)
        )
