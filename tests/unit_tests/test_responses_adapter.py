import json
from typing import Any

import httpx
import pytest
import respx
from openai.types.responses import Response, ResponseUsage
from openai.types.responses.response_usage import (
    InputTokensDetails,
    OutputTokensDetails,
)

from aidial_adapter_openai.responses.converter import (
    chat_completions_to_responses_request,
)
from tests.utils.mock_server import MockServer

_UPSTREAM_ENDPOINT = "https://api.openai.com/v1/responses"
_INPUT_TOKENS_ENDPOINT = f"{_UPSTREAM_ENDPOINT}/input_tokens"
_UNSUPPORTED_RESPONSES_UPSTREAM_ENDPOINTS = [
    "https://test.openai.azure.com/openai/v1/responses",
    "https://test.services.ai.azure.com/openai/v1/responses",
    "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
]


def _response(usage: ResponseUsage | None = None) -> Response:
    return Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=usage,
    )


def _headers(upstream_endpoint: str = _UPSTREAM_ENDPOINT) -> dict[str, str]:
    return {
        "X-UPSTREAM-KEY": "test-api-key",
        "X-UPSTREAM-ENDPOINT": upstream_endpoint,
    }


def _chat_completion_chunks(response: httpx.Response) -> list[dict[str, Any]]:
    chunks = []
    for line in response.text.splitlines():
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        chunks.append(json.loads(line.removeprefix("data: ")))
    return chunks


@respx.mock
async def test_response_model_name(test_app: httpx.AsyncClient):
    def check_request(request: httpx.Request):
        assert json.loads(request.content)["model"] == "upstream-model-name"
        return httpx.Response(
            status_code=200, content=json.dumps(_response().model_dump())
        )

    respx.post(_UPSTREAM_ENDPOINT).mock(side_effect=check_request)

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "messages": [{"role": "user", "content": "Test content"}],
        },
        headers=_headers(),
    )

    assert response.status_code == 200


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_max_prompt_tokens_truncates_messages(
    test_app: httpx.AsyncClient, stream: bool
):
    token_counts = iter([100, 80, 40])
    create_bodies: list[dict[str, Any]] = []

    @respx.post(_INPUT_TOKENS_ENDPOINT)
    def _count_tokens(request: httpx.Request):
        return httpx.Response(
            status_code=200,
            json={
                "object": "response.input_tokens",
                "input_tokens": next(token_counts),
            },
        )

    @MockServer().post(_UPSTREAM_ENDPOINT)
    def _create_response(request: httpx.Request):
        create_bodies.append(json.loads(request.content))
        return (
            MockServer.mock_responses_api_response("text.txt")
            if stream
            else _response()
        )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": stream,
            "max_prompt_tokens": 50,
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ],
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    assert [body["input"] for body in create_bodies] == [
        [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "new question"},
        ]
    ]

    if stream:
        chunks = _chat_completion_chunks(response)
        assert chunks[-1]["statistics"] == {"discarded_messages": [1, 2]}
    else:
        assert response.json()["statistics"] == {"discarded_messages": [1, 2]}


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    "upstream_endpoint", _UNSUPPORTED_RESPONSES_UPSTREAM_ENDPOINTS
)
async def test_max_prompt_tokens_ignored_for_unsupported_responses_vendor(
    test_app: httpx.AsyncClient, stream: bool, upstream_endpoint: str
):
    create_bodies: list[dict[str, Any]] = []

    @MockServer().post(upstream_endpoint)
    def _create_response(request: httpx.Request):
        create_bodies.append(json.loads(request.content))
        return (
            MockServer.mock_responses_api_response("text.txt")
            if stream
            else _response()
        )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": stream,
            "max_prompt_tokens": 50,
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ],
        },
        headers=_headers(upstream_endpoint),
    )

    assert response.status_code == 200
    assert [body["input"] for body in create_bodies] == [
        [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ]
    ]

    if stream:
        chunks = _chat_completion_chunks(response)
        assert "statistics" not in chunks[-1]
    else:
        assert "statistics" not in response.json()


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_chat_completions_without_max_prompt_tokens_does_not_truncate(
    test_app: httpx.AsyncClient, stream: bool
):
    create_bodies: list[dict[str, Any]] = []

    @MockServer().post(_UPSTREAM_ENDPOINT)
    def _create_response(request: httpx.Request):
        create_bodies.append(json.loads(request.content))
        return (
            MockServer.mock_responses_api_response("text.txt")
            if stream
            else _response()
        )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": stream,
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "old question"},
                {"role": "assistant", "content": "old answer"},
                {"role": "user", "content": "new question"},
            ],
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    assert [body["input"] for body in create_bodies] == [
        [
            {"role": "system", "content": "system message"},
            {"role": "user", "content": "old question"},
            {"role": "assistant", "content": "old answer"},
            {"role": "user", "content": "new question"},
        ]
    ]

    if stream:
        chunks = _chat_completion_chunks(response)
        assert "statistics" not in chunks[-1]
    else:
        assert "statistics" not in response.json()


@respx.mock
async def test_usage_reports_cache_tokens(test_app: httpx.AsyncClient):
    usage = ResponseUsage(
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        input_tokens_details=InputTokensDetails(
            cached_tokens=30, cache_write_tokens=70
        ),
        output_tokens_details=OutputTokensDetails(reasoning_tokens=5),
    )

    @MockServer().post(_UPSTREAM_ENDPOINT)
    def _create_response(request: httpx.Request):
        return _response(usage)

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "messages": [{"role": "user", "content": "Test content"}],
        },
        headers=_headers(),
    )

    assert response.status_code == 200
    assert response.json()["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "total_tokens": 120,
        "prompt_tokens_details": {
            "audio_tokens": None,
            "cached_tokens": 30,
            "cache_write_tokens": 70,
        },
        "completion_tokens_details": {
            "accepted_prediction_tokens": None,
            "audio_tokens": None,
            "reasoning_tokens": 5,
            "rejected_prediction_tokens": None,
        },
    }


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
