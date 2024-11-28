import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from aidial_adapter_openai.app import create_app
from aidial_adapter_openai.utils.request import get_app_config


@pytest.fixture
def _app_instance():
    return create_app(init_telemetry=False)


@pytest_asyncio.fixture
async def test_app(_app_instance):
    async with httpx.AsyncClient(
        transport=ASGITransport(app=_app_instance),
        base_url="http://test-app.com",
    ) as client:
        yield client


@pytest.fixture
def eliminate_empty_choices(_app_instance):
    app_config = get_app_config(_app_instance)
    app_config.ELIMINATE_EMPTY_CHOICES = True
    yield
    app_config.ELIMINATE_EMPTY_CHOICES = False
