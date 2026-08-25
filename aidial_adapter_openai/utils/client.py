from anthropic import AsyncAnthropic
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

import aidial_adapter_openai.providers.alibaba as alibaba
from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    DeploymentAPIType,
)
from aidial_adapter_openai.utils.auth import get_credentials

_Client = AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI | AsyncAnthropic


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
    extra_vendor_headers = alibaba.get_extra_headers(
        vendor, deployment.deployment_type
    )

    return deployment_endpoint.get_client(
        {
            **creds,
            "api_version": api_version,
            "headers": extra_headers | extra_vendor_headers,
        }
    )
