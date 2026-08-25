import json

import httpx
import pytest
import respx
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.dial_api.normalize_request import (
    normalize_dial_request,
)
from tests.utils.stream import OpenAIStream, single_choice_chunk

_EXPLICIT = {"prompt_cache_breakpoint": {"mode": "explicit"}}


def test_cache_breakpoint_on_string_content():
    request = {
        "messages": [
            {
                "role": "user",
                "content": "hi",
                "custom_fields": {"cache_breakpoint": {}},
            }
        ]
    }

    assert normalize_dial_request(request) == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi", **_EXPLICIT}],
            }
        ]
    }


def test_cache_breakpoint_marks_last_content_part():
    request = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "a"},
                    {"type": "text", "text": "b"},
                ],
                "custom_fields": {"cache_breakpoint": {"expire_at": "later"}},
            }
        ]
    }

    assert normalize_dial_request(request)["messages"] == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "a"},
                {"type": "text", "text": "b", **_EXPLICIT},
            ],
        }
    ]


def test_cache_breakpoint_without_content_is_dropped():
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [],
                "custom_fields": {"cache_breakpoint": {}},
            }
        ]
    }

    assert normalize_dial_request(request)["messages"] == [
        {"role": "assistant", "content": None, "tool_calls": []}
    ]


def test_dial_fields_are_dropped():
    request = {
        "custom_fields": {"configuration": {"foo": "bar"}},
        "messages": [
            {
                "role": "user",
                "content": "hi",
                "custom_content": {
                    "state": {"a": 1},
                    "stages": [],
                    "attachments": [],
                },
                "custom_fields": {"unknown": 1},
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f"},
                "custom_fields": {"cache_breakpoint": {}},
            }
        ],
    }

    assert normalize_dial_request(request) == {
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }


def test_request_is_not_mutated():
    message = {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
        "custom_fields": {"cache_breakpoint": {}},
    }
    request = {"messages": [message]}

    normalize_dial_request(request)

    assert request == {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hi"}],
                "custom_fields": {"cache_breakpoint": {}},
            }
        ]
    }


def test_malformed_dial_field_is_rejected():
    request = {"messages": [{"role": "user", "custom_content": "oops"}]}

    with pytest.raises(InvalidRequestError):
        normalize_dial_request(request)


_API_VERSION = "api-version=2023-03-15-preview"
_UPSTREAM_ENDPOINT = (
    "http://localhost:5001/openai/deployments/gpt-4/chat/completions"
)


@respx.mock
async def test_dial_fields_are_not_proxied_upstream(
    test_app: httpx.AsyncClient,
):
    upstream_body: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal upstream_body
        upstream_body = json.loads(request.content)
        return httpx.Response(
            status_code=200,
            json=OpenAIStream(
                single_choice_chunk(
                    finish_reason="stop",
                    delta={"role": "assistant", "content": "hello"},
                )
            ).to_block_response(),
        )

    respx.post(f"{_UPSTREAM_ENDPOINT}?{_API_VERSION}").mock(side_effect=handler)

    response = await test_app.post(
        f"/openai/deployments/gpt-4/chat/completions?{_API_VERSION}",
        json={
            "stream": False,
            "custom_fields": {"configuration": {"foo": "bar"}},
            "messages": [
                {
                    "role": "user",
                    "content": "hi",
                    "custom_fields": {"cache_breakpoint": {}},
                    "custom_content": {"state": {"a": 1}},
                }
            ],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200
    assert "custom_fields" not in upstream_body
    assert upstream_body["messages"] == [
        {
            "role": "user",
            "content": [{"type": "text", "text": "hi", **_EXPLICIT}],
        }
    ]
