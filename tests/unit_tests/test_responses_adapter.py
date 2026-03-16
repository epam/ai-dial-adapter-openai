import json

import httpx
import respx
from openai.types.responses import Response

from tests.conftest import AzureOpenAIClientFactory
from tests.utils.mock_server import MockServer


@respx.mock
async def test_response_model_name(
    create_azure_openai_client: AzureOpenAIClientFactory,
):
    upstream_endpoint = "http://localhost:5001/openai/v1/responses"
    upstream_model_name = "test-upstream-model-name"

    @MockServer().post(upstream_endpoint)
    def _responses(request: httpx.Request):
        model = json.loads(request.content)["model"]
        assert model == upstream_model_name
        return Response(
            id="id",
            created_at=0,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )

    client = create_azure_openai_client(
        "test-deployment-name", upstream_endpoint=upstream_endpoint
    )

    response = await client.chat.completions.create(
        model=upstream_model_name,
        messages=[{"role": "user", "content": "Test content"}],
    )

    assert response.choices[0].message.content == ""
