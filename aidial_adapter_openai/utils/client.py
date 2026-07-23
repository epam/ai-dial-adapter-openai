from anthropic import AsyncAnthropicFoundry
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    DeploymentAPIType,
)
from aidial_adapter_openai.utils.auth import get_credentials

_Client = (
    AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI | AsyncAnthropicFoundry
)


async def get_client(
    request: Request,
    deployment_id: str | None,
    deployment: DeploymentAPIType,
    app_config: ApplicationConfig,
    extra_headers: dict[str, str],
    api_version: str | None,
) -> _Client:
    deployment_endpoint = deployment.endpoint
    vendor = app_config.get_vendor(deployment_id, deployment_endpoint)
    creds = await get_credentials(request.headers, vendor=vendor)
    return deployment_endpoint.get_client(
        {**creds, "api_version": api_version, "headers": extra_headers}
    )
