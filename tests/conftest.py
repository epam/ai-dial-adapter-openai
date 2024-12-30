import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncAzureOpenAI

from aidial_adapter_openai.app import create_app
from aidial_adapter_openai.utils.http_client import DEFAULT_TIMEOUT
from aidial_adapter_openai.utils.request import get_app_config
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG


@pytest.fixture(scope="session")
def _app_instance():
    return create_app(
        init_telemetry=False,
        app_config=TEST_DEPLOYMENTS_CONFIG.app_config,
    )


@pytest.fixture
async def test_app(_app_instance):
    async with httpx.AsyncClient(
        transport=ASGITransport(app=_app_instance),
        base_url="http://test-app.com",
        timeout=DEFAULT_TIMEOUT,
    ) as client:
        yield client


@pytest.fixture
def eliminate_empty_choices(_app_instance):
    app_config = get_app_config(_app_instance)
    app_config.ELIMINATE_EMPTY_CHOICES = True
    yield
    app_config.ELIMINATE_EMPTY_CHOICES = False


@pytest.fixture
def get_openai_client(test_app: httpx.AsyncClient):
    def _get_client(deployment_config: DeploymentConfig) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_app.base_url),
            azure_deployment=deployment_config.deployment_id,
            api_version="2024-02-01",
            api_key="dummy_key",
            max_retries=5,
            timeout=30,
            http_client=test_app,
            default_headers=deployment_config.upstream_headers,
        )

    yield _get_client
