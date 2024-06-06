from aidial_sdk.exceptions import HTTPException as DialException
from fastapi import Response
from fastapi.responses import StreamingResponse
from openai import Completion
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.env import COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES
from aidial_adapter_openai.utils.auth import get_auth_header
from aidial_adapter_openai.utils.exceptions import handle_exceptions
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import map_stream


def convert_completion_to_chat_completion(item: OpenAIObject, role="assistant"):
    item_dict = item.to_dict_recursive()
    return {
        "object": "chat.completion",
        "id": item_dict["id"],
        "model": item_dict["model"],
        "created": item_dict["created"],
        "usage": item_dict["usage"],
        "choices": [
            {
                "index": choice["index"],
                "finish_reason": choice["finish_reason"],
                "message": {
                    "content": choice["text"],
                    "role": role,
                },
                "logprobs": choice["logprobs"],
            }
            for choice in item_dict["choices"]
        ],
    }


async def legacy_completions(
    data,
    deployment_id,
    is_stream,
    api_key,
    api_type,
    api_version,
    api_base,
    model=None,
    engine=None,
):

    if data.get("n", 1) > 1:
        raise DialException(
            status_code=422,
            message="The deployment doesn't support n > 1",
            type="invalid_request_error",
        )
    messages = data.get("messages", [])

    if len(messages) == 0:
        raise DialException(
            status_code=422,
            message="The request doesn't contain any messages",
            type="invalid_request_error",
        )

    prompt = messages[-1].get("content") or ""
    if (
        template := COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES.get(
            data.get("deployment_id")
        )
    ) is not None:
        prompt = template.format(prompt=prompt)

    response = await handle_exceptions(
        Completion.acreate(
            headers={
                **get_auth_header(api_type=api_type, api_key=api_key),
            },
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            api_key=api_key,
            model=data.get("model") or model,
            prompt=prompt,
            stream=is_stream,
            deployment_id=deployment_id if api_type == "azure" else None,
            engine=engine,
            timeout=DEFAULT_TIMEOUT,
        )
    )
    if isinstance(response, Response):
        return response

    if is_stream:
        chunks_formatted = map_stream(
            lambda obj: convert_completion_to_chat_completion(obj), response
        )
        return StreamingResponse(
            to_openai_sse_stream(chunks_formatted),
            media_type="text/event-stream",
        )
    else:
        return convert_completion_to_chat_completion(response)
