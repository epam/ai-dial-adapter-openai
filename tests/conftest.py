from unittest.mock import patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport

from aidial_adapter_openai.app import app


@pytest.fixture
def eliminate_empty_choices():
    with patch(
        "aidial_adapter_openai.utils.streaming.ELIMINATE_EMPTY_CHOICES", True
    ):
        yield


@pytest_asyncio.fixture
async def test_app():
    async with httpx.AsyncClient(
        transport=ASGITransport(app=app),  # type: ignore
        base_url="http://test-app.com",
    ) as client:
        yield client
