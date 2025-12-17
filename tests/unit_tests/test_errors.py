import json
from typing import Any, AsyncIterable, AsyncIterator, Callable
from unittest.mock import patch

import httpx
import pytest
import respx
from respx.types import SideEffectTypes

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.conftest import create_test_client
from tests.integration_tests.constants import (
    IMAGE_RESOURCE,
    PDF_DOCUMENT_RESOURCE,
)
from tests.utils.dictionary import exclude_keys
from tests.utils.openai import (
    user_with_file_content_part,
    user_with_image_content_part,
)
from tests.utils.stream import (
    OpenAIStream,
    create_choice,
    many_choices_chunk,
    single_choice_chunk,
)


def assert_equal(actual: Any, expected: Any):
    assert actual == expected


def assert_equal_no_dynamic_fields(actual: Any, expected: Any):
    if isinstance(actual, dict) and isinstance(expected, dict):
        keys = {"id", "created"}
        assert exclude_keys(actual, keys) == exclude_keys(expected, keys)
    else:
        assert actual == expected


def mock_response(
    status_code: int,
    content_type: str,
    content: str,
    *,
    check_request: Callable[[httpx.Request], None] = lambda _: None,
    extra_headers: dict[str, str] = {},
) -> SideEffectTypes:
    def side_effect(request: httpx.Request):
        check_request(request)
        return httpx.Response(
            status_code=status_code,
            headers={
                "content-type": content_type,
                **extra_headers,
            },
            content=content,
        )

    return side_effect


@respx.mock
async def test_single_chunk_token_counting(test_app: httpx.AsyncClient):
    # The adapter tolerates top-level extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "5"}, finish_reason="stop"
        ),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
            }
        },
    )


@respx.mock
async def test_top_level_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates top-level extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {"error": {"message": "whatever", "code": "500"}}
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            content_type="text/event-stream",
            content=mock_stream.to_content(),
            check_request=check_request,
        ),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "extra_field": 1,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(response, assert_equal)


@respx.mock
async def test_nested_extra_field(test_app: httpx.AsyncClient):
    # The adapter tolerates nested extra fields
    # and passes it further to the upstream endpoint.

    mock_stream = OpenAIStream(
        {"error": {"message": "whatever", "code": "500"}}
    )

    def check_request(request: httpx.Request):
        assert json.loads(request.content)["messages"][0]["extra_field"] == 1

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            status_code=200,
            content_type="text/event-stream",
            content=mock_stream.to_content(),
            check_request=check_request,
        ),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [
                {"role": "user", "content": "2+3=?", "extra_field": 1}
            ],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(response, assert_equal)


@respx.mock
async def test_missing_api_version(test_app: httpx.AsyncClient):

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "400",
            "message": "api-version is a required query parameter",
            "type": "invalid_request_error",
        }
    }


@respx.mock
async def test_error_during_streaming_stopped(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(finish_reason="stop", delta={"role": "assistant"}),
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
                "code": "500",
                "extra_error_field": "extra_error_value",
            }
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "prompt_tokens": 9,
                "completion_tokens": 0,
                "total_tokens": 9,
            }
        },
    )


@respx.mock
async def test_error_during_streaming_unfinished(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hello "}),
        {
            "error": {
                "message": "Error test",
                "type": "runtime_error",
                "code": "500",
                "extra_error_field": "extra_error_value",
            }
        },
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal,
        usages={
            0: {
                "completion_tokens": 2,
                "prompt_tokens": 9,
                "total_tokens": 11,
            }
        },
    )


@respx.mock
async def test_interrupted_stream_single_choice(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant", "content": "hello"}),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200

    expected_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "hello"},
            finish_reason="length",
            usage={
                "completion_tokens": 1,
                "prompt_tokens": 9,
                "total_tokens": 10,
            },
        )
    )
    expected_stream.assert_response_content(response, assert_equal)


@respx.mock
async def test_interrupted_stream_many_choices(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "hello1"}, choice_index=0
        ),
        single_choice_chunk(
            delta={"role": "assistant", "content": "hello2"}, choice_index=1
        ),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "n": 3,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200

    expected_stream = OpenAIStream(
        single_choice_chunk(
            choice_index=0,
            delta={"role": "assistant", "content": "hello1"},
        ),
        many_choices_chunk(
            choices=[
                create_choice(
                    index=1,
                    delta={"role": "assistant", "content": "hello2"},
                    finish_reason="length",
                ),
                create_choice(index=0, finish_reason="length"),
                create_choice(index=2, finish_reason="length"),
            ],
            usage={
                "completion_tokens": 4,
                "prompt_tokens": 9,
                "total_tokens": 13,
            },
        ),
    )
    expected_stream.assert_response_content(response, assert_equal)


@respx.mock
async def test_zero_chunk_stream(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream()

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200

    expected_final_chunk = single_choice_chunk(
        delta={},
        finish_reason="length",
        usage={"prompt_tokens": 9, "completion_tokens": 0, "total_tokens": 9},
    )

    expected_stream = OpenAIStream(expected_final_chunk)
    expected_stream.assert_response_content(
        response, assert_equal_no_dynamic_fields
    )


@respx.mock
async def test_incorrect_upstream_url(test_app: httpx.AsyncClient):
    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            # upstream endpoint should contain the full path
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001",
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Invalid upstream endpoint format",
            "type": "invalid_request_error",
            "code": "400",
        }
    }


@respx.mock
async def test_no_request_response_validation(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200, json={"messages": "string", "extra_response": "string"}
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Test content",
                    "extra_mesage": "string",
                }
            ],
            "extra_request": "string",
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
            "Content-Type": "application/pdf",
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "messages": "string",
        "extra_response": "string",
    }


@respx.mock
async def test_status_error_from_upstream(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(status_code=400, content="Bad request")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 400
    assert response.text == "Bad request"


@respx.mock
async def test_status_error_from_upstream_with_headers(
    test_app: httpx.AsyncClient,
):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=429,
        content="Too many requests",
        headers={"Retry-After": "42"},
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 429
    assert response.text == "Too many requests"
    assert response.headers["Retry-After"] == "42"


@respx.mock
async def test_timeout_error_from_upstream(test_app: httpx.AsyncClient):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(side_effect=httpx.ReadTimeout("Timeout error"))

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 504
    assert response.json() == {
        "error": {
            "message": "Request timed out",
            "type": "timeout",
            "code": "504",
            "display_message": "Request timed out. Please try again later.",
        }
    }


@respx.mock
async def test_connection_error_from_upstream_non_streaming(
    test_app: httpx.AsyncClient,
):
    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(side_effect=httpx.ConnectError("Connection error"))

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 502
    assert response.json() == {
        "error": {
            "message": "Error communicating with OpenAI",
            "type": "connection",
            "code": "502",
            "display_message": "OpenAI server is not responsive. Please try again later.",
        }
    }


@respx.mock
async def test_content_length_of_response_error(test_app: httpx.AsyncClient):
    upstream_response = """
{
    "error": {
        "message": "Bad request",

        "code": "400"

    }
}
"""
    upstream_response_content_length = str(len(upstream_response))

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).mock(
        side_effect=mock_response(
            400,
            "application/json",
            upstream_response,
            extra_headers={"content-length": upstream_response_content_length},
        )
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={"messages": [{"role": "user", "content": "Test content"}]},
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    expected_response = json.dumps(
        json.loads(upstream_response), separators=(",", ":")
    )
    expected_content_length = str(len(expected_response))

    assert response.status_code == 400
    assert response.text == expected_response
    assert response.headers["content-length"] == expected_content_length
    assert upstream_response_content_length != expected_content_length


@respx.mock
async def test_connection_error_from_upstream_streaming(
    test_app: httpx.AsyncClient,
):
    async def mock_stream() -> AsyncIterable[bytes]:
        yield b'data: {"message": "first chunk"}\n\n'
        yield b'data: {"message": "second chunk"}\n\n'
        raise httpx.ConnectError("Connection error")

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Test content"}],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    assert response.text == "\n\n".join(
        [
            'data: {"message":"first chunk"}',
            'data: {"message":"second chunk"}',
            'data: {"error":{"message":"Connection error","type":"internal_server_error","code":"500"}}',
            "data: [DONE]",
            "",
        ]
    )


@respx.mock
async def test_adapter_internal_error(
    test_app: httpx.AsyncClient,
):
    async def mock_generate_stream(stream: AsyncIterator[dict], **kwargs):
        yield await stream.__anext__()
        raise ValueError("failed generating the stream")

    with patch(
        "aidial_adapter_openai.chat_completions.gpt.generate_stream",
        side_effect=mock_generate_stream,
    ):

        async def mock_stream() -> AsyncIterable[bytes]:
            yield b'data: {"message": "first chunk"}\n\n'
            yield b'data: {"message": "second chunk"}\n\n'
            yield b"data: [DONE]"

        respx.post(
            "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
        ).respond(
            status_code=200,
            content_type="text/event-stream",
            content=mock_stream(),
        )

        response = await test_app.post(
            "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
            json={
                "stream": True,
                "messages": [{"role": "user", "content": "Test content"}],
            },
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
            },
        )

        assert response.status_code == 200
        assert response.text == "\n\n".join(
            [
                'data: {"message":"first chunk"}',
                'data: {"error":{"message":"failed generating the stream","type":"internal_server_error","code":"500"}}',
                "data: [DONE]",
                "",
            ]
        )


@respx.mock
async def test_invalid_chunk_stream_from_upstream(
    test_app: httpx.AsyncClient,
):
    async def mock_stream() -> AsyncIterable[bytes]:
        yield b"data: chunk1\n\n"
        yield b"data: chunk2\n\n"
        yield b"data: [DONE]\n\n"

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream(),
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Test content"}],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    assert response.status_code == 200
    assert response.text == "\n\n".join(
        [
            # OpenAI is unable to parse SSE entry with invalid JSON and fails with the following error:
            'data: {"error":{"message":"Expecting value: line 1 column 1 (char 0)","type":"internal_server_error","code":"500"}}',
            "data: [DONE]",
            "",
        ]
    )


@respx.mock
async def test_unexpected_multi_modal_input_streaming(
    test_app: httpx.AsyncClient, caplog
):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test response"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "stream": True,
            "messages": [
                user_with_image_content_part("image1", IMAGE_RESOURCE),
                user_with_image_content_part("image2", IMAGE_RESOURCE),
                user_with_file_content_part(
                    "file1", "file1", PDF_DOCUMENT_RESOURCE
                ),
            ],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    file_not_supported = (
        "Content part type 'file' is not supported by the tokenizer. "
        "Tokens for this content part will be ignored."
    )

    image_not_supported = (
        "Image content detected, however, the image tokenization algorithm is not known for this deployment. "
        "Tokens for the image will be ignored. "
        "Declare the deployment in either GPT4O_DEPLOYMENTS or GPT4O_MINI_DEPLOYMENTS "
        "to specify the image tokenization algorithm."
    )

    log_messages = [record.message for record in caplog.records]
    assert file_not_supported in log_messages
    assert image_not_supported in log_messages

    assert response.status_code == 200
    mock_stream.assert_response_content(
        response,
        assert_equal_no_dynamic_fields,
        usages={
            2: {
                "prompt_tokens": 21,
                "completion_tokens": 2,
                "total_tokens": 23,
            }
        },
    )


@respx.mock
async def test_invalid_image_url_streaming_catch_all(
    test_app: httpx.AsyncClient,
):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test response"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
        content_type="text/event-stream",
    )

    image_url = "http://xyz.com/image.png"

    respx.get(image_url).respond(status_code=404, content="Not Found")

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": image_url,
                                "detail": "auto",
                            },
                        }
                    ],
                }
            ],
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    error_message = f"The following files failed to process:\n1. {image_url}: failed to download the image content part"

    response_stream = OpenAIStream(
        {
            "error": {
                "code": "400",
                "type": "invalid_request_error",
                "message": error_message,
                "display_message": error_message,
            }
        },
    )

    assert response.status_code == 400
    response_stream.assert_response_content(
        response, assert_equal_no_dynamic_fields
    )


@respx.mock
async def test_invalid_image_url_streaming_gpt4o():
    app_config = (
        ApplicationConfig()
        .add_deployment("app", ChatCompletionDeploymentType.GPT4O)
        .map_to_tiktoken_model("app", "gpt-4")
    )

    async with create_test_client(app_config) as test_app:
        mock_stream = OpenAIStream(
            single_choice_chunk(delta={"role": "assistant"}),
            single_choice_chunk(delta={"content": "Test response"}),
            single_choice_chunk(delta={}, finish_reason="stop"),
        )

        respx.post(
            "http://localhost:5001/openai/deployments/upstream-model/chat/completions?api-version=2023-03-15-preview"
        ).respond(
            status_code=200,
            content=mock_stream.to_content(),
            content_type="text/event-stream",
        )

        image_url = "http://xyz.com/image.png"

        respx.get(image_url).respond(status_code=404, content="Not Found")

        response = await test_app.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={
                "stream": True,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                    "detail": "auto",
                                },
                            }
                        ],
                    }
                ],
            },
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/upstream-model/chat/completions",
            },
        )

        error_message = f"The following files failed to process:\n1. {image_url}: failed to download the image content part"

        response_stream = OpenAIStream(
            {
                "error": {
                    "code": "400",
                    "type": "invalid_request_error",
                    "message": error_message,
                    "display_message": error_message,
                }
            },
        )

        assert response.status_code == 400
        response_stream.assert_response_content(
            response, assert_equal_no_dynamic_fields
        )


async def test_incorrect_max_prompt_tokens_streaming_request(
    test_app: httpx.AsyncClient,
):
    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
            "max_prompt_tokens": 0,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    expected_response = {
        "error": {
            "code": "400",
            "message": "'0' is less than the minimum of 1",
            "type": "invalid_request_error",
            "param": "max_prompt_tokens",
        }
    }

    assert response.status_code == 400
    assert response.json() == expected_response


@respx.mock
@pytest.mark.parametrize("stream", [False, True])
async def test_error_from_gpt_multi_modal(stream: bool):
    app_config = (
        ApplicationConfig()
        .add_deployment("app", ChatCompletionDeploymentType.GPT4O)
        .map_to_tiktoken_model("app", "gpt-4")
    )

    upstream_url = "http://test-upstream/openai/deployments/upstream-deployment/chat/completions"

    respx.post(f"{upstream_url}?api-version=2023-03-15-preview").respond(
        status_code=500,
        content="Something went wrong",
        content_type="text/plain",
    )

    async with create_test_client(app_config=app_config) as http_client:
        response = await http_client.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [{"role": "user", "content": "test"}],
                "stream": stream,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_url,
            },
        )

        assert response.status_code == 500
        assert response.content == b"Something went wrong"


@respx.mock
@pytest.mark.parametrize(
    "no_usage_returned", [True, False], ids=["no-usage", "with-usage"]
)
@pytest.mark.parametrize(
    "with_max_prompt_tokens",
    [True, False],
    ids=["with-truncate-prompt", "no-truncate-prompt"],
)
async def test_missing_tiktoken_model(
    test_app: httpx.AsyncClient,
    no_usage_returned: bool,
    with_max_prompt_tokens: bool,
    caplog,
):
    usage_from_upstream = (
        None
        if no_usage_returned
        else {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3}
    )

    mock_stream = OpenAIStream(
        single_choice_chunk(
            delta={"role": "assistant", "content": "5"},
            finish_reason="stop",
            usage=usage_from_upstream,
        ),
    )

    respx.post(
        "http://test-upstream/openai/deployments/upstream-deployment/chat/completions?api-version=2023-03-15-preview"
    ).respond(
        status_code=200,
        content_type="text/event-stream",
        content=mock_stream.to_content(),
    )

    response = await test_app.post(
        "/openai/deployments/my-favorite-model/chat/completions?api-version=2023-03-15-preview",
        json={
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
            **({"max_prompt_tokens": 100} if with_max_prompt_tokens else {}),
        },
        headers={
            "X-UPSTREAM-KEY": "dummy-upstream-api-key",
            "X-UPSTREAM-ENDPOINT": "http://test-upstream/openai/deployments/upstream-deployment/chat/completions",
        },
    )

    assert response.status_code == 200

    warnings = [
        record.message
        for record in caplog.records
        if record.levelname == "WARNING"
    ]

    if no_usage_returned or with_max_prompt_tokens:
        # The adapter-side tokenization is only needed when either
        # * the usage isn't provided by the upstream, or
        # * "max_prompt_tokens" parameter was in the request.
        tiktoken_warning = """
    Could not find tokenizer for the model 'my-favorite-model' in the tiktoken package. Consider mapping the model to an existing tokenizer via TIKTOKEN_MODEL_MAPPING variable in the adapter OpenAI environment: TIKTOKEN_MODEL_MAPPING='{"my-favorite-model": $prefix}', where $prefix is one of: "o1-", "o3-", "o4-mini-", "gpt-5-", "gpt-4.5-", "gpt-4.1-", "chatgpt-4o-", "gpt-4o-", "gpt-4-", "gpt-3.5-turbo-", "gpt-35-turbo-", "gpt-oss-". Alternatively, declare the deployment as a model that doesn't require tokenization via tiktoken. Meantime, the default tokenizer of the 'gpt-4o' model will be used instead: 'o200k_base'.
    """.strip()

        assert len(warnings) == 1
        assert tiktoken_warning == warnings[0]
    else:
        assert warnings == []


@pytest.mark.parametrize("stream", [False, True])
async def test_error_invalid_image_url(stream: bool):
    app_config = (
        ApplicationConfig()
        .add_deployment("app", ChatCompletionDeploymentType.GPT4O)
        .map_to_tiktoken_model("app", "gpt-4")
    )

    upstream_url = "http://test-upstream/openai/deployments/upstream-deployment/chat/completions"

    async with create_test_client(app_config=app_config) as http_client:
        response = await http_client.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "test"},
                            {"type": "image_url", "image_url": "whatever"},
                        ],
                    }
                ],
                "stream": stream,
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_url,
            },
        )

        assert response.status_code == 400
        assert response.json() == {
            "error": {
                "message": "'image_url' expected to be dict, but got str",
                "type": "invalid_request_error",
                "code": "400",
            }
        }


@respx.mock
async def test_rate_limit_exceeded_during_streaming():
    app_config = (
        ApplicationConfig()
        .add_deployment("app", ChatCompletionDeploymentType.RESPONSES_API)
        .map_to_tiktoken_model("app", "gpt-4")
    )

    upstream_url = "http://test-upstream.com/openai/v1/responses"

    mock_event = {
        "type": "response.in_progress",
        "sequence_number": 1,
        "response": {
            "id": "resp_01f342feea0be5f60069419a50a74c81908afe72661bbd3112",
            "created_at": 1765907024.0,
            "metadata": {},
            "model": "gpt-5.2-2025-12-11",
            "object": "response",
            "output": [],
            "parallel_tool_calls": True,
            "temperature": 1.0,
            "tool_choice": "auto",
            "tools": [],
            "top_p": 0.98,
            "background": False,
            "reasoning": {
                "effort": "none",
            },
            "service_tier": "auto",
            "status": "in_progress",
            "text": {"format": {"type": "text"}, "verbosity": "medium"},
            "truncation": "disabled",
            "store": True,
            "top_logprobs": 0,
        },
    }

    mock_stream = OpenAIStream(
        mock_event,
        {
            "error": {
                "message": "no_kv_space",
                "type": "server_error",
                "code": "rate_limit_exceeded",
            }
        },
    )

    respx.post("http://test-upstream.com/openai/v1/responses").mock(
        side_effect=mock_response(
            status_code=200,
            content_type="text/event-stream",
            content=mock_stream.to_content(),
        )
    )

    async with create_test_client(app_config=app_config) as http_client:
        response = await http_client.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={
                "model": "upstream-model-id",
                "messages": [{"role": "user", "content": "test"}],
                "stream": "True",
            },
            headers={
                "X-UPSTREAM-KEY": "dummy-upstream-api-key",
                "X-UPSTREAM-ENDPOINT": upstream_url,
            },
        )

        assert response.status_code == 500
        assert response.json() == {
            "error": {
                "code": "rate_limit_exceeded",
                "message": "no_kv_space",
                "type": "server_error",
            }
        }
