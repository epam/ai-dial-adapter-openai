import fastapi
from aidial_adapter_anthropic.adapter import ChatCompletionAdapter, UserError
from aidial_adapter_anthropic.adapter.claude import ApproximateTokenizer
from aidial_adapter_anthropic.adapter.claude import (
    create_adapter as create_anthropic_adapter,
)
from aidial_adapter_anthropic.dial.consumer import ChoiceConsumer
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_adapter_anthropic.dial.storage import FileStorage
from aidial_sdk.chat_completion import Request as DIALRequest
from aidial_sdk.chat_completion import Response as DIALResponse
from anthropic import AsyncAnthropicFoundry
from fastapi.responses import StreamingResponse

from aidial_adapter_openai.dial_api.sdk_adapter import sdk_adapter
from aidial_adapter_openai.dial_api.storage import DIAL_URL
from aidial_adapter_openai.utils.env import get_env_int

_CLAUDE_DEFAULT_MAX_TOKENS = get_env_int("CLAUDE_DEFAULT_MAX_TOKENS", 1536)


def _create_file_storage(api_key: str | None) -> FileStorage | None:
    if api_key is None or DIAL_URL is None:
        return None

    return FileStorage(dial_url=DIAL_URL, api_key=api_key)


async def _create_adapter(
    model_name: str, api_key: str, client: AsyncAnthropicFoundry
) -> ChatCompletionAdapter:
    return await create_anthropic_adapter(
        deployment=model_name,
        storage=_create_file_storage(api_key),
        client=client,
        custom_tokenizer=ApproximateTokenizer(),
        default_max_tokens=_CLAUDE_DEFAULT_MAX_TOKENS,
        supports_thinking=True,
        supports_documents=True,
    )


async def chat_completion(
    *,
    request: fastapi.Request,
    deployment_id: str,
    model_name: str,
    client: AsyncAnthropicFoundry,
) -> StreamingResponse | dict:
    async def _handler(request: DIALRequest, response: DIALResponse) -> None:
        model = await _create_adapter(model_name, request.api_key, client)
        response.set_model(model_name)

        params = ModelParameters.create(request)

        async with ChoiceConsumer(response) as consumer:
            try:
                await model.chat(consumer, params, request.messages)
            except UserError as e:
                await e.report_usage(consumer.choice)
                await response.aflush()
                raise e

    return await sdk_adapter(
        request=request,
        deployment_id=deployment_id,
        chat_completion=_handler,
    )
