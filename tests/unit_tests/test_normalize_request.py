import json

import httpx
import pytest
import respx
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.dial_api.normalize_request import (
    DialFeatures,
    _parse_dial_request,
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


def test_cache_breakpoint_on_refusal_part_is_dropped():
    request = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "refusal", "refusal": "no"}],
                "custom_fields": {"cache_breakpoint": {}},
            }
        ]
    }

    assert normalize_dial_request(request)["messages"] == [
        {
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "no"}],
        }
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


def test_parsing_collects_every_dial_field():
    request = {
        "custom_fields": {"configuration": {"foo": "bar"}},
        "messages": [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": "hi",
                "custom_content": {"state": {"a": 1}},
                "custom_fields": {"cache_breakpoint": {}},
            },
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "f"},
                "custom_fields": {"cache_breakpoint": {}},
            }
        ],
    }

    parsed, features = _parse_dial_request(request)

    assert parsed == {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ],
        "tools": [{"type": "function", "function": {"name": "f"}}],
    }
    assert features == DialFeatures(
        custom_fields={"configuration": {"foo": "bar"}},
        message_custom_content={1: {"state": {"a": 1}}},
        message_custom_fields={1: {"cache_breakpoint": {}}},
        tool_custom_fields={0: {"cache_breakpoint": {}}},
    )


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


def test_static_tool_is_rejected():
    request = {
        "messages": [],
        "tools": [
            {"type": "function", "function": {"name": "f"}},
            {"type": "static_function", "static_function": {"name": "g"}},
        ],
    }

    with pytest.raises(InvalidRequestError, match=r"tools\[1\]"):
        normalize_dial_request(request)


def test_malformed_dial_field_is_rejected():
    request = {"messages": [{"role": "user", "custom_fields": "oops"}]}

    with pytest.raises(InvalidRequestError):
        normalize_dial_request(request)


def test_unexpected_error_leaves_the_request_intact():
    request = {"messages": 42, "custom_fields": {"configuration": {}}}

    assert normalize_dial_request(request) is request


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
