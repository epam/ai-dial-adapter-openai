import contextlib

import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncAzureOpenAI

from aidial_adapter_openai.app import create_app
from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.request import get_app_config
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG

_TEST_TIMEOUT = httpx.Timeout(30, connect=10)


@pytest.fixture(autouse=True)
async def disable_caches():
    # Disable all caches that may retain references to event loops.
    #
    # pytest-asyncio creates a new event loop for each test with scope="function".
    # If a cached object is created or a global object is constructed in an early test,
    # it may hold a reference to that test’s event loop.
    #
    # Once that loop is closed and a new one is created for the next test,
    # the cached object becomes invalid — leading to the error:
    # "RuntimeError: Event loop is closed"
    #
    # To avoid this, we clear or disable any caches that may hold event loop-bound state.

    # Disable `functools.cache` caches
    from aidial_adapter_openai.utils.http_client import get_http_client

    get_http_client.cache_clear()


@pytest.fixture(scope="session")
def _app_instance():
    return create_app(
        init_telemetry=False,
        app_config=TEST_DEPLOYMENTS_CONFIG.app_config,
    )


@pytest.fixture
async def test_app(_app_instance):
    async with httpx.AsyncClient(
        transport=ASGITransport(app=_app_instance, raise_app_exceptions=False),
        base_url="http://test-app.com",
        timeout=_TEST_TIMEOUT,
    ) as client:
        yield client


@pytest.fixture
def eliminate_empty_choices(_app_instance):
    app_config = get_app_config(_app_instance)
    app_config.ELIMINATE_EMPTY_CHOICES = True
    yield
    app_config.ELIMINATE_EMPTY_CHOICES = False


@contextlib.asynccontextmanager
async def create_test_client(
    app_config: ApplicationConfig, *, base_url: str = "http://test-app.com"
):
    app = create_app(init_telemetry=False, app_config=app_config)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore
        base_url=base_url,
        timeout=_TEST_TIMEOUT,
    ) as client:
        yield client


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
