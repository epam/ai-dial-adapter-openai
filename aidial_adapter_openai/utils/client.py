from anthropic import AsyncAnthropicFoundry
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    DeploymentAPIType,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.request import get_api_version

_Client = (
    AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI | AsyncAnthropicFoundry
)


async def get_client(
    request: Request,
    deployment_id: str | None,
    deployment: DeploymentAPIType,
    app_config: ApplicationConfig,
    extra_headers: dict[str, str],
) -> _Client:
    deployment_endpoint = deployment.endpoint
    vendor = app_config.get_vendor(deployment_id, deployment_endpoint)
    creds = await get_credentials(request.headers, vendor=vendor)
    api_version = get_api_version(request)
    return deployment_endpoint.get_client(
        {**creds, "api_version": api_version, "headers": extra_headers}
    )
