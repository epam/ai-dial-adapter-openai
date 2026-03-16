import json

import httpx
import respx
from openai.types.responses import Response

from tests.conftest import OpenAIClientFactory
from tests.utils.mock_server import MockServer

_UPSTREAM_MODEL = "test-upstream-model-name"
_TEST_RESPONSE = Response(
    id="id",
    created_at=0,
    model=_UPSTREAM_MODEL,
    object="response",
    output=[],
    parallel_tool_calls=False,
    tool_choice="none",
    tools=[],
).model_dump()


@respx.mock
async def test_responses_endpoint(create_openai_client: OpenAIClientFactory):
    upstream_endpoint = "http://localhost:5001/openai/v1/responses"

    @MockServer().post(upstream_endpoint)
    def _responses(request: httpx.Request):
        model = json.loads(request.content)["model"]
        assert model == _UPSTREAM_MODEL
        return _TEST_RESPONSE

    client = create_openai_client(upstream_endpoint=upstream_endpoint)

    response = await client.responses.create(
        model=_UPSTREAM_MODEL, input="Test content"
    )

    assert response.output == []


@respx.mock
async def test_responses_endpoint_proxy_response_headers(
    create_openai_client: OpenAIClientFactory,
):
    upstream_endpoint = "http://localhost:5001/openai/v1/responses"

    @MockServer().post(upstream_endpoint)
    def _responses(request: httpx.Request):
        model = json.loads(request.content)["model"]
        assert model == _UPSTREAM_MODEL
        return httpx.Response(
            status_code=200,
            json=_TEST_RESPONSE,
            headers={"foo": "bar"},
        )

    client = create_openai_client(upstream_endpoint=upstream_endpoint)

    response = await client.responses.with_raw_response.create(
        model=_UPSTREAM_MODEL, input="Test content"
    )

    assert response.status_code == 200
    assert response.headers.get("foo") == "bar"
