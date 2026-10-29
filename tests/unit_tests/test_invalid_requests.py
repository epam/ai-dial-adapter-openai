"""
Malformed requests are rejected with 400 before reaching the upstream,
rather than crashing the adapter with 500.
"""

import httpx
import pytest
import respx

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from tests.conftest import create_test_client

_GPT = "http://localhost:5001/openai/deployments/gpt-4/chat/completions"
_RESPONSES = "https://api.openai.com/v1/responses"
_ANTHROPIC = "https://api.anthropic.com/v1/messages"


def _user(content: object) -> dict:
    return {"role": "user", "content": content}


@pytest.mark.parametrize(
    ("upstream_endpoint", "body", "message"),
    [
        (_GPT, {}, "Missing required parameter: 'messages'"),
        (
            _GPT,
            {"messages": None},
            "'messages' expected to be list, but got null",
        ),
        (
            _GPT,
            {"messages": [{"content": "2+3=?"}]},
            "Missing required parameter: 'messages[0].role'",
        ),
        (
            _RESPONSES,
            {"messages": [{"role": "userr", "content": "2+3=?"}]},
            "Invalid value of 'messages[0].role': 'userr'. Supported values: "
            "'system', 'developer', 'user', 'assistant', 'tool', 'function'",
        ),
        (
            _RESPONSES,
            {"messages": [{"role": None, "content": "2+3=?"}]},
            "'messages[0].role' expected to be str, but got null",
        ),
        (
            _GPT,
            {"messages": [_user([{"type": "video_url"}])]},
            "Invalid value of 'messages[0].content[0].type': 'video_url'. "
            "Supported values: 'image_url', 'input_audio', 'text', 'refusal', 'file'",
        ),
        (
            _GPT,
            {"messages": [_user([{"text": "2+3=?"}])]},
            "Missing required parameter: 'messages[0].content[0].type'",
        ),
        (
            _GPT,
            {"messages": [_user([{"type": "text"}])]},
            "Missing required parameter: 'messages[0].content[0].text'",
        ),
        (
            _GPT,
            {"messages": [_user([{"type": "text", "text": None}])]},
            "'messages[0].content[0].text' expected to be str, but got null",
        ),
        (
            _GPT,
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "2+3=?",
                        "custom_content": {"attachments": [{"title": "x"}]},
                    }
                ]
            },
            "Invalid request. Path: 'messages[0].custom_content.attachments[0]', "
            "error: Value error, "
            "Attachment must have either 'data' or 'url', but it's missing both",
        ),
        (
            _RESPONSES,
            {
                "messages": [
                    _user("2+3=?"),
                    {
                        "role": "assistant",
                        "tool_calls": [{"type": "function", "function": {}}],
                    },
                ]
            },
            "Missing required parameter: 'messages[1].tool_calls[0].id'",
        ),
        (
            _RESPONSES,
            {"messages": [_user("2+3=?")], "tool_choice": {"type": "foo"}},
            "Invalid value of 'tool_choice.type': 'foo'. "
            "Supported values: 'allowed_tools', 'function', 'custom'",
        ),
        (
            _RESPONSES,
            {
                "messages": [_user("2+3=?")],
                "response_format": {"json_schema": {}},
            },
            "Missing required parameter: 'response_format.type'",
        ),
        (
            _ANTHROPIC,
            {"messages": [{"role": "userr", "content": "2+3=?"}]},
            "Invalid request. Path: 'messages[0].role', error: Input should be "
            "'system', 'developer', 'user', 'assistant', 'function' or 'tool'",
        ),
        (
            _GPT,
            {
                "messages": [_user("2+3=?")],
                "max_prompt_tokens": 100,
                "tools": ["x"],
            },
            "'tools[0]' expected to be dict, but got str",
        ),
        (
            _RESPONSES,
            {
                "messages": [_user("2+3=?")],
                "tools": [{"type": "static_function", "static_function": "x"}],
            },
            "'tools[0].static_function' expected to be dict, but got str",
        ),
        (
            _RESPONSES,
            {
                "messages": [_user("2+3=?")],
                "tools": [
                    {
                        "type": "static_function",
                        "static_function": {
                            "name": "web_search",
                            "configuration": "x",
                        },
                    }
                ],
            },
            "'tools[0].static_function.configuration' expected to be dict, "
            "but got str",
        ),
        (
            _GPT,
            {"messages": [_user("2+3=?")], "extra_body": "x", "foo": 1},
            "'extra_body' expected to be dict, but got str",
        ),
    ],
    ids=[
        "no-messages",
        "null-messages",
        "no-role",
        "unknown-role",
        "null-role",
        "unknown-content-part-type",
        "no-content-part-type",
        "no-text",
        "null-text",
        "attachment-without-data-or-url",
        "tool-call-without-id",
        "unknown-tool-choice-type",
        "response-format-without-type",
        "sdk-request-validation",
        "tool-to-tokenize-not-dict",
        "static-function-not-dict",
        "static-function-configuration-not-dict",
        "extra-body-not-dict",
    ],
)
@respx.mock
async def test_chat_completion(
    test_app: httpx.AsyncClient,
    upstream_endpoint: str,
    body: dict,
    message: str,
):
    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2024-02-01",
        json={**body, "stream": True},
        headers={
            "Api-Key": "TEST_API_KEY",
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": upstream_endpoint,
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": message,
            "type": "invalid_request_error",
            "code": "400",
        }
    }


@respx.mock
async def test_embeddings():
    upstream_endpoint = "http://localhost:5001/v1/embeddings"
    app_config = ApplicationConfig(VLLM_DEPLOYMENTS=["qwen3-vl-embed"])

    async with create_test_client(app_config) as client:
        response = await client.post(
            "/openai/deployments/qwen3-vl-embed/embeddings?api-version=2024-02-01",
            json={"input": 1},
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": upstream_endpoint,
            },
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Invalid request. Path: 'input.str', "
            "error: Input should be a valid string",
            "type": "invalid_request_error",
            "code": "400",
        }
    }
