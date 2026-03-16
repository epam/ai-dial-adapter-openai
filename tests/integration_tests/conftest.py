import httpx
import pytest
from openai import AsyncAzureOpenAI

from tests.integration_tests.base import DeploymentConfig


@pytest.fixture
def create_openai_client(test_app: httpx.AsyncClient):
    def _create_client(deployment_config: DeploymentConfig) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_app.base_url),
            azure_deployment=deployment_config.id_,
            api_version="2024-12-01-preview",
            api_key="dummy_key",
            max_retries=3,
            http_client=test_app,
            default_headers=deployment_config.upstream_headers,
        )

    yield _create_client
