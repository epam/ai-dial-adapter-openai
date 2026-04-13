import contextlib
from typing import Generator, Protocol

import httpx
import pytest
from httpx import ASGITransport
from openai import AsyncAzureOpenAI, AsyncOpenAI

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.request import get_app_config
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG

_TEST_TIMEOUT = httpx.Timeout(30, connect=10)


@pytest.fixture()
def _app_instance():
    from aidial_adapter_openai.app import create_app

    return create_app(
        init_telemetry=False,
        app_config=TEST_DEPLOYMENTS_CONFIG.app_config,
    )


@pytest.fixture
async def test_app(_app_instance):
    async with httpx.AsyncClient(
        transport=ASGITransport(app=_app_instance),
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
    from aidial_adapter_openai.app import create_app

    app = create_app(init_telemetry=False, app_config=app_config)
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app, raise_app_exceptions=False),  # type: ignore
        base_url=base_url,
        timeout=_TEST_TIMEOUT,
    ) as client:
        yield client


class AzureOpenAIClientFactory(Protocol):
    def __call__(
        self,
        azure_deployment: str,
        *,
        max_retries: int | None = None,
        api_version: str | None = None,
        upstream_endpoint: str | None = None,
        upstream_key: str | None = None,
    ) -> AsyncAzureOpenAI: ...


@pytest.fixture
def create_azure_openai_client(
    test_app: httpx.AsyncClient,
) -> Generator[AzureOpenAIClientFactory, None, None]:
    def _create_client(
        azure_deployment: str,
        *,
        max_retries: int | None = None,
        api_version: str | None = None,
        upstream_endpoint: str | None = None,
        upstream_key: str | None = "test-upstream-api-key",
    ) -> AsyncAzureOpenAI:
        default_headers: dict[str, str] = {}
        if upstream_key is not None:
            default_headers["X-UPSTREAM-KEY"] = upstream_key
        if upstream_endpoint is not None:
            default_headers["X-UPSTREAM-ENDPOINT"] = upstream_endpoint

        return AsyncAzureOpenAI(
            azure_endpoint=str(test_app.base_url),
            http_client=test_app,
            azure_deployment=azure_deployment,
            api_key="test-adapter-api-key",
            api_version=api_version or "2024-12-01-preview",
            max_retries=max_retries or 0,
            default_headers=default_headers,
        )

    yield _create_client


class OpenAIClientFactory(Protocol):
    def __call__(
        self,
        *,
        max_retries: int | None = None,
        upstream_endpoint: str | None = None,
        upstream_key: str | None = None,
    ) -> AsyncOpenAI: ...


@pytest.fixture
def create_openai_client(
    test_app: httpx.AsyncClient,
) -> Generator[OpenAIClientFactory, None, None]:
    def _create_client(
        *,
        max_retries: int | None = None,
        upstream_endpoint: str | None = None,
        upstream_key: str | None = "test-upstream-api-key",
    ) -> AsyncOpenAI:
        default_headers: dict[str, str] = {}
        if upstream_key is not None:
            default_headers["X-UPSTREAM-KEY"] = upstream_key
        if upstream_endpoint is not None:
            default_headers["X-UPSTREAM-ENDPOINT"] = upstream_endpoint

        return AsyncOpenAI(
            base_url=f"{str(test_app.base_url)}/openai/v1",
            http_client=test_app,
            api_key="test-adapter-api-key",
            max_retries=max_retries or 0,
            default_headers=default_headers,
        )

    yield _create_client
