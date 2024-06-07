from aidial_sdk.exceptions import HTTPException as DialException
from fastapi import Response
from fastapi.responses import StreamingResponse
from openai import Completion
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.env import COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES
from aidial_adapter_openai.types import ChatCompletionRequestData
from aidial_adapter_openai.utils.auth import get_auth_header
from aidial_adapter_openai.utils.exceptions import handle_exceptions
from aidial_adapter_openai.utils.remove_none import remove_none
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import map_stream


def convert_completion_to_chat_completion(
    item: OpenAIObject, role="assistant", is_stream: bool = False
):
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
                ("message" if is_stream else "delta"): {
                    "content": choice["text"],
                    "role": role,
                },
                "logprobs": choice["logprobs"],
            }
            for choice in item_dict["choices"]
        ],
    }


async def legacy_completions(
    data: ChatCompletionRequestData,
    deployment_id: str,
    is_stream: bool,
    api_key: str,
    api_type: str,
    api_version: str,
    api_base: str,
    model=None,
    engine=None,
):
    if data.get("n", 1) > 1:  # type: ignore
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
        template := COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES.get(deployment_id)
    ) is not None:
        prompt = template.format(prompt=prompt)
    extra_params = {
        "frequency_penalty": data.get("frequency_penalty"),
        "best_of": data.get("best_of"),
        "logit_bias": data.get("logit_bias"),
        "max_tokens": data.get("max_tokens"),
        "top_p": data.get("top_p"),
        "temperature": data.get("temperature"),
        "presence_penalty": data.get("presence_penalty"),
        "stop": data.get("stop"),
        "engine": data.get("engine"),
    }
    response = await handle_exceptions(
        Completion.acreate(
            headers={
                **get_auth_header(api_type=api_type, api_key=api_key),
            },
            api_base=api_base,
            api_version=api_version,
            api_type=api_type,
            api_key=api_key,
            prompt=prompt,
            stream=is_stream,
            deployment_id=deployment_id if api_type == "azure" else None,
            model=data.get("model") or model,
            timeout=DEFAULT_TIMEOUT,
            **remove_none(extra_params),
        )
    )
    if isinstance(response, Response):
        return response

    if is_stream:
        chunks_formatted = map_stream(
            lambda obj: convert_completion_to_chat_completion(
                obj, is_stream=is_stream
            ),
            response,
        )
        return StreamingResponse(
            to_openai_sse_stream(chunks_formatted),
            media_type="text/event-stream",
        )
    else:
        return convert_completion_to_chat_completion(response)
