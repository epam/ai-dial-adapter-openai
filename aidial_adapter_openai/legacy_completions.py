from aidial_sdk.exceptions import HTTPException as DialException
from openai import Completion

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.env import (
    DEPLOYMENT_TO_PROMPT_PROCESSOR,
)
from aidial_adapter_openai.utils.string_processor import (
    process_string,
)


async def legacy_completions(
    data,
    deployment_id,
    is_stream,
    upstream_endpoint,
    api_key,
    api_type,
    api_version,
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

    if prompt_processor := DEPLOYMENT_TO_PROMPT_PROCESSOR.get(deployment_id):
        prompt = process_string(prompt, prompt_processor)

    return await Completion.acreate(
        api_base=upstream_endpoint,
        api_version=api_version,
        api_type=api_type,
        api_key=api_key,
        model=data["model"],
        prompt=prompt,
        stream=is_stream,
        timeout=DEFAULT_TIMEOUT,
    )
