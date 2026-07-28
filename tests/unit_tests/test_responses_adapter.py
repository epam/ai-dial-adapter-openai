import json

import httpx
import respx
from openai.types.responses import Response

from aidial_adapter_openai.responses.converter import (
    chat_completions_to_responses_request,
)


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
            status_code=200, content=json.dumps(dummy_response.model_dump())
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


async def test_assistant_message_content_parts():
    _, create_request = await chat_completions_to_responses_request(
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Hi"},
                        {"type": "refusal", "refusal": "No way"},
                    ],
                },
            ],
        },
        file_storage=None,
    )

    assert create_request.get("input") == [
        {
            "role": "user",
            "content": [{"type": "input_text", "text": "Hello"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "output_text", "text": "Hi", "annotations": []},
                {"type": "refusal", "refusal": "No way"},
            ],
        },
    ]
