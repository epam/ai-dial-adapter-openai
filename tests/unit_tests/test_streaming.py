import httpx
import respx

from tests.utils.stream import OpenAIStream, chunk, single_choice_chunk


def assert_equal(actual, expected):
    assert actual == expected


@respx.mock
async def test_streaming_computed_tokens(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test content"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
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
            2: {
                "completion_tokens": 2,
                "prompt_tokens": 9,
                "total_tokens": 11,
            }
        },
    )


@respx.mock
async def test_streaming_inherited_tokens(test_app: httpx.AsyncClient):
    mock_stream = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test content"}),
        single_choice_chunk(
            delta={},
            finish_reason="stop",
            usage={
                "completion_tokens": 111,
                "prompt_tokens": 222,
                "total_tokens": 333,
            },
        ),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
        content=mock_stream.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
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
    mock_stream.assert_response_content(response, assert_equal)


@respx.mock
async def test_include_usage(test_app: httpx.AsyncClient):
    # Emulating the steam returned by OpenAI when stream_options.include_usage=True
    upstream_response = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test content"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
        chunk(
            choices=[],
            usage={
                "completion_tokens": 111,
                "prompt_tokens": 222,
                "total_tokens": 333,
            },
        ),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
        content=upstream_response.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
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
    upstream_response.assert_response_content(response, assert_equal)


@respx.mock
async def test_include_usage_stream_issues(
    eliminate_empty_choices, test_app: httpx.AsyncClient
):
    # Emulating the steam returned by OpenAI when stream_options.include_usage=True
    upstream_response = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test content"}),
        single_choice_chunk(delta={}, finish_reason="stop"),
        chunk(
            choices=[],
            usage={
                "completion_tokens": 111,
                "prompt_tokens": 222,
                "total_tokens": 333,
            },
        ),
    )

    respx.post(
        "http://localhost:5001/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15"
    ).respond(
        status_code=200,
        content=upstream_response.to_content(),
        content_type="text/event-stream",
    )

    response = await test_app.post(
        "/openai/deployments/gpt-4/chat/completions?api-version=2023-06-15",
        json={
            "messages": [{"role": "user", "content": "Test content"}],
            "stream": True,
        },
        headers={
            "X-UPSTREAM-KEY": "TEST_API_KEY",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/gpt-4/chat/completions",
        },
    )

    expected_response = OpenAIStream(
        single_choice_chunk(delta={"role": "assistant"}),
        single_choice_chunk(delta={"content": "Test content"}),
        single_choice_chunk(
            delta={},
            finish_reason="stop",
            usage={
                "completion_tokens": 111,
                "prompt_tokens": 222,
                "total_tokens": 333,
            },
        ),
    )

    assert response.status_code == 200
    expected_response.assert_response_content(response, assert_equal)
