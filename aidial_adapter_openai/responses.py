from typing import Any, AsyncIterator, Dict

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)


async def chat_completion(
    data: Dict[str, Any],
    endpoint: OpenAIEndpoint | AzureOpenAIEndpoint,
    creds: OpenAICreds,
    is_stream: bool,
    api_version: str,
    deployment_id: str,
    app_config: ApplicationConfig,
) -> AsyncIterator[dict] | dict: ...
