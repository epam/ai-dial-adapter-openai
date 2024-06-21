import httpx
import pytest_asyncio
from httpx import ASGITransport

from aidial_adapter_openai.app import app


@pytest_asyncio.fixture
async def test_app():
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore
        base_url="http://test-app.com",
    ) as client:
        yield client
