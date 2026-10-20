"""
The upstream client is created per request, so the signature cache must be
keyed by something that outlives it.
"""

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.reflection import _method_signature
from tests.conftest import create_test_client
from tests.utils.stream import OpenAIStream, single_choice_chunk

_DEPLOYMENT = "gpt-test"
_UPSTREAM = "http://test-upstream/v1/chat/completions"


@pytest.fixture
async def client():
    async with create_test_client(
        app_config=ApplicationConfig(
            TIKTOKEN_MODEL_MAPPING={_DEPLOYMENT: "gpt-4o"}
        )
    ) as client:
        yield client


@respx.mock
async def test_signature_cache_is_shared_between_requests(
    client: httpx.AsyncClient,
):
    respx.post(_UPSTREAM).respond(
        json=OpenAIStream(
            single_choice_chunk(
                delta={"role": "assistant", "content": "5"},
                finish_reason="stop",
            )
        ).to_block_response()
    )
    _method_signature.cache_clear()

    for _ in range(2):
        response = await client.post(
            f"openai/deployments/{_DEPLOYMENT}/chat/completions",
            params={"api-version": "2024-02-01"},
            json={"messages": [{"role": "user", "content": "2+3=?"}]},
            headers={
                "Api-Key": "test-adapter-api-key",
                "X-UPSTREAM-KEY": "test-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": _UPSTREAM,
            },
        )
        assert response.status_code == 200

    assert _method_signature.cache_info().currsize == 1
