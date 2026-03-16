import json

import httpx
import respx
from openai.types.responses import Response

from tests.conftest import OpenAIClientFactory
from tests.utils.mock_server import MockServer


@respx.mock
async def test_response_endpoint(create_openai_client: OpenAIClientFactory):
    upstream_endpoint = "http://localhost:5001/openai/v1/responses"
    upstream_model_name = "test-upstream-model-name"

    @MockServer().post(upstream_endpoint)
    def _responses(request: httpx.Request):
        model = json.loads(request.content)["model"]
        assert model == upstream_model_name

        return Response(
            id="id",
            created_at=0,
            model=model,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    client = create_openai_client(upstream_endpoint=upstream_endpoint)

    response = await client.responses.create(
        model=upstream_model_name, input="Test content"
    )

    assert response.output == []
