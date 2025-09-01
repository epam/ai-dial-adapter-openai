import json

import httpx
import respx
from openai.types.responses import Response


@respx.mock
async def test_response_model_name(test_app: httpx.AsyncClient):
    def check_request(request: httpx.Request):
        assert json.loads(request.content)["model"] == "upstream-model-name"
        dummy_response = Response(
            id="id",
            created_at=0,
            model="test-model",
            object="response",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        return httpx.Response(
            status_code=200, content=json.dumps(dummy_response.dict())
        )

    respx.post("http://localhost:5001/openai/v1/responses").mock(
        side_effect=check_request
    )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "messages": [{"role": "user", "content": "Test content"}],
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/v1/responses",
        },
    )

    assert response.status_code == 200
