import json
from typing import Any

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client
from tests.utils.mock_server import MockServer
from tests.utils.stream import OpenAIStream, single_choice_chunk

_CHAT_COMPLETIONS_UPSTREAM = "http://test-upstream/v1/chat/completions"
_RESPONSES_UPSTREAM = "http://test-upstream/v1/responses"

# Matched by the "ali.*" glob pattern in ALIBABA_DEPLOYMENTS.
_GLOB_DEPLOYMENT = "ali.model-name"
_LISTED_DEPLOYMENT = "test-alibaba-model-name"

_SESSION_CACHE_HEADER = "x-dashscope-session-cache"
_IMAGE_URL = (
    "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
    "AAAADUlEQVR42mP8z8AAAwAB/gEBmVRxCwAAAABJRU5ErkJggg=="
)
_BREAKPOINT_PATH_HEADER = "X-DIAL-CACHE-BREAKPOINT-PATH"


@pytest.fixture
async def test_app():
    app_config = ApplicationConfig(
        ALIBABA_DEPLOYMENTS=["ali.*", _LISTED_DEPLOYMENT],
        GPT4O_DEPLOYMENTS=["ali.*"],
    )
    async with create_test_client(app_config=app_config) as client:
        yield client


def _mock_chat_completions_upstream() -> None:
    respx.post(_CHAT_COMPLETIONS_UPSTREAM).respond(
        status_code=200,
        content_type="application/json",
        content=json.dumps(
            OpenAIStream(
                single_choice_chunk(
                    delta={"role": "assistant", "content": "5"},
                    finish_reason="stop",
                )
            ).to_block_response()
        ),
    )


def _mock_responses_upstream() -> None:
    MockServer().post(_RESPONSES_UPSTREAM)(
        MockServer.mock_responses_api_response("text.txt")
    )


async def _post_chat_completions(
    test_app: httpx.AsyncClient,
    *,
    deployment_id: str,
    messages: list[dict[str, Any]],
    upstream_endpoint: str = _CHAT_COMPLETIONS_UPSTREAM,
    extra_headers: dict[str, str] | None = None,
    **extra_body: Any,
) -> httpx.Response:
    return await test_app.post(
        f"/openai/deployments/{deployment_id}/chat/completions"
        "?api-version=2024-02-01",
        json={"model": "model-name", "messages": messages, **extra_body},
        headers={
            "X-UPSTREAM-KEY": "test-upstream-api-key",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            **(extra_headers or {}),
        },
    )


def _upstream_request() -> httpx.Request:
    return respx.calls.last.request


def _upstream_messages() -> Any:
    return json.loads(_upstream_request().content)["messages"]


_CACHE_BREAKPOINT = {"custom_fields": {"cache_breakpoint": {}}}
_CACHE_CONTROL = {"cache_control": {"type": "ephemeral"}}
_MESSAGE_WITH_BREAKPOINT = {
    "role": "user",
    "content": "2+3=?",
    **_CACHE_BREAKPOINT,
}


@respx.mock
async def test_cache_breakpoints_are_converted(test_app: httpx.AsyncClient):
    _mock_chat_completions_upstream()

    response = await _post_chat_completions(
        test_app,
        deployment_id=_GLOB_DEPLOYMENT,
        messages=[
            {
                "role": "system",
                "content": "be a helpful assistant",
                **_CACHE_BREAKPOINT,
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "the manual"},
                    {"type": "text", "text": "2+3=?"},
                ],
                **_CACHE_BREAKPOINT,
            },
            {"role": "assistant", "content": "5"},
            {"role": "user", "content": "and 3+4=?"},
        ],
    )

    assert response.status_code == 200
    assert _upstream_messages() == [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "be a helpful assistant",
                    **_CACHE_CONTROL,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "the manual"},
                {"type": "text", "text": "2+3=?", **_CACHE_CONTROL},
            ],
        },
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "and 3+4=?"},
    ]
    # The session cache is only requested for the Responses API deployments
    assert _SESSION_CACHE_HEADER not in _upstream_request().headers


@respx.mock
async def test_cache_breakpoints_are_converted_for_listed_deployment(
    test_app: httpx.AsyncClient,
):
    _mock_chat_completions_upstream()

    response = await _post_chat_completions(
        test_app,
        deployment_id=_LISTED_DEPLOYMENT,
        messages=[_MESSAGE_WITH_BREAKPOINT],
    )

    assert response.status_code == 200
    assert _upstream_messages() == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "2+3=?", **_CACHE_CONTROL}],
        }
    ]


@respx.mock
async def test_top_level_cache_breakpoint_marks_the_last_message(
    test_app: httpx.AsyncClient,
):
    _mock_chat_completions_upstream()

    response = await _post_chat_completions(
        test_app,
        deployment_id=_GLOB_DEPLOYMENT,
        messages=[
            {"role": "system", "content": "be a helpful assistant"},
            {"role": "user", "content": "2+3=?"},
        ],
        custom_fields=_CACHE_BREAKPOINT["custom_fields"],
    )

    assert response.status_code == 200
    request = json.loads(_upstream_request().content)
    assert "custom_fields" not in request
    assert request["messages"] == [
        {"role": "system", "content": "be a helpful assistant"},
        {
            "role": "user",
            "content": [{"type": "text", "text": "2+3=?", **_CACHE_CONTROL}],
        },
    ]


@respx.mock
@pytest.mark.parametrize("breakpoint_message_index", [0, 3, None])
async def test_reported_dial_breakpoint_path_covers_the_whole_prompt(
    test_app: httpx.AsyncClient, breakpoint_message_index: int | None
):
    _mock_chat_completions_upstream()

    # The path is only reported above the caching token threshold
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": "word " * 2000},
        {"role": "user", "content": "2+3=?"},
        {"role": "assistant", "content": "5"},
        {"role": "user", "content": "and 3+4=?"},
    ]
    if breakpoint_message_index is not None:
        messages[breakpoint_message_index] |= _CACHE_BREAKPOINT

    response = await _post_chat_completions(
        test_app,
        deployment_id=_GLOB_DEPLOYMENT,
        messages=messages,
        # DIAL Core sends the header for an auto-caching deployment
        extra_headers={_BREAKPOINT_PATH_HEADER: "whatever"},
    )

    assert response.status_code == 200
    assert (
        response.headers[_BREAKPOINT_PATH_HEADER] == "prefix.body.messages[3]"
    )


@respx.mock
@pytest.mark.parametrize(
    "message",
    [
        pytest.param(
            {
                "role": "user",
                "content": "what is on the picture?",
                "custom_content": {
                    "attachments": [{"type": "image/png", "url": _IMAGE_URL}]
                },
            },
            id="dial-attachment",
        ),
        pytest.param(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what is on the picture?"},
                    {"type": "image_url", "image_url": {"url": _IMAGE_URL}},
                ],
            },
            id="content-parts",
        ),
    ],
)
async def test_cache_breakpoint_marks_last_content_part(
    test_app: httpx.AsyncClient, message: dict[str, Any]
):
    _mock_chat_completions_upstream()

    response = await _post_chat_completions(
        test_app,
        # An image-capable deployment served by Model Studio
        deployment_id=_GLOB_DEPLOYMENT,
        messages=[message | _CACHE_BREAKPOINT],
    )

    assert response.status_code == 200
    content = _upstream_messages()[0]["content"]
    assert [part["type"] for part in content] == ["text", "image_url"]
    assert "cache_control" not in content[0]
    assert content[-1]["cache_control"] == _CACHE_CONTROL["cache_control"]


@respx.mock
async def test_cache_breakpoints_are_intact_for_non_alibaba_deployment(
    test_app: httpx.AsyncClient,
):
    _mock_chat_completions_upstream()

    response = await _post_chat_completions(
        test_app,
        deployment_id="gpt-4o",
        messages=[_MESSAGE_WITH_BREAKPOINT],
    )

    assert response.status_code == 200
    assert _upstream_messages() == [_MESSAGE_WITH_BREAKPOINT]


@respx.mock
@pytest.mark.parametrize(
    ("deployment_id", "expected_header"),
    [
        (_GLOB_DEPLOYMENT, "enable"),
        (_LISTED_DEPLOYMENT, "enable"),
        ("gpt-4o", None),
    ],
)
async def test_session_cache_header_in_responses_adapter(
    test_app: httpx.AsyncClient,
    deployment_id: str,
    expected_header: str | None,
):
    _mock_responses_upstream()

    response = await _post_chat_completions(
        test_app,
        deployment_id=deployment_id,
        messages=[{"role": "user", "content": "2+3=?"}],
        upstream_endpoint=_RESPONSES_UPSTREAM,
    )

    assert response.status_code == 200
    assert (
        _upstream_request().headers.get(_SESSION_CACHE_HEADER)
        == expected_header
    )


@respx.mock
@pytest.mark.parametrize(
    ("model", "expected_header"),
    [(_GLOB_DEPLOYMENT, "enable"), ("gpt-4o", None)],
)
async def test_session_cache_header_in_responses_passthrough(
    test_app: httpx.AsyncClient, model: str, expected_header: str | None
):
    _mock_responses_upstream()

    response = await test_app.post(
        "/openai/v1/responses",
        json={"model": model, "input": "2+3=?"},
        headers={
            "X-UPSTREAM-KEY": "test-upstream-api-key",
            "X-UPSTREAM-ENDPOINT": _RESPONSES_UPSTREAM,
        },
    )

    assert response.status_code == 200
    assert (
        _upstream_request().headers.get(_SESSION_CACHE_HEADER)
        == expected_header
    )
