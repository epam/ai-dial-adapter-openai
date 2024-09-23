import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

import aidial_adapter_openai.utils.streaming
from aidial_adapter_openai.app import app


@pytest.fixture
def eliminate_empty_choices(monkeypatch):
    monkeypatch.setattr(
        aidial_adapter_openai.utils.streaming, "eliminate_empty_choices", True
    )


@pytest_asyncio.fixture
async def test_app():

    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore
        base_url="http://test-app.com",
    ) as client:
        yield client
