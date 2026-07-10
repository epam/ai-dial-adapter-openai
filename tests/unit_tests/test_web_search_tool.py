import json

import httpx
import pytest
import respx
from aidial_sdk.exceptions import RequestValidationError
from openai.types.responses import Response
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ActionSearchSource,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from aidial_adapter_openai.responses.converter import (
    convert_response,
    convert_tools,
)


def _response_output_message(text: str) -> ResponseOutputMessage:
    return ResponseOutputMessage(
        id="msg_id",
        type="message",
        role="assistant",
        status="completed",
        content=[
            ResponseOutputText(
                type="output_text",
                text=text,
                annotations=[],
            )
        ],
    )


def _response_web_search(item_id: str, query: str) -> ResponseFunctionWebSearch:
    return ResponseFunctionWebSearch(
        id=item_id,
        type="web_search_call",
        status="completed",
        # openai==2.16.0 still validates deprecated `query` as required.
        action=ActionSearch(type="search", query=query, queries=[query]),
    )


def _to_sse_content(events: list[dict]) -> str:
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + (
        "data: [DONE]\n\n"
    )


def _read_stream_chunks(response: httpx.Response) -> tuple[list[dict], bool]:
    chunks = []
    saw_done = False
    for line in response.iter_lines():
        if not line:
            continue
        assert line.startswith("data: ")
        payload = line.removeprefix("data: ")
        if payload == "[DONE]":
            saw_done = True
            continue
        chunks.append(json.loads(payload))
    return chunks, saw_done


@pytest.mark.parametrize(
    ("request_tool", "expected_tool"),
    [
        (
            {
                "type": "static_function",
                "static_function": {"name": "web_search"},
            },
            {"type": "web_search"},
        ),
        (
            {
                "type": "static_function",
                "static_function": {
                    "name": "web_search",
                    "configuration": {"search_context_size": "high"},
                },
            },
            {"type": "web_search", "search_context_size": "high"},
        ),
    ],
)
@respx.mock
async def test_web_search_tool_conversion(
    test_app: httpx.AsyncClient,
    request_tool: dict,
    expected_tool: dict,
):
    def check_request(request: httpx.Request):
        body = json.loads(request.content)
        assert body["tools"] == [expected_tool]
        dummy_response = Response(
            id="id",
            created_at=0,
            model="test-model",
            object="response",
            output=[_response_output_message("The weather in Kyiv is sunny.")],
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
            "messages": [
                {"role": "user", "content": "What is the weather in Kyiv?"}
            ],
            "tools": [request_tool],
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/v1/responses",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert (
        body["choices"][0]["message"]["content"]
        == "The weather in Kyiv is sunny."
    )


@respx.mock
async def test_web_search_streaming_annotations_and_attachments(
    test_app: httpx.AsyncClient,
):
    def check_request(request: httpx.Request):
        body = json.loads(request.content)
        assert body["tools"] == [
            {
                "type": "web_search",
                "user_location": {"city": "London", "type": "approximate"},
            }
        ]

        created_response = Response(
            id="resp_id",
            created_at=0,
            model="test-model",
            object="response",
            status="in_progress",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        completed_response = Response(
            id="resp_id",
            created_at=0,
            model="test-model",
            object="response",
            status="completed",
            output=[],
            parallel_tool_calls=False,
            tool_choice="none",
            tools=[],
        )
        events = [
            {
                "type": "response.created",
                "response": created_response.model_dump(),
            },
            {
                "type": "response.web_search_call.in_progress",
                "item_id": "msg_id",
                "output_index": 0,
            },
            {
                "item": {
                    "id": "msg_id",
                    "action": None,
                    "status": "in_progress",
                    "type": "web_search_call",
                },
                "output_index": 0,
                "sequence_number": 2,
                "type": "response.output_item.added",
            },
            {
                "item": {
                    "id": "msg_id",
                    "action": {
                        "queries": ["Weather in Kyiv"],
                        "sources": [
                            {
                                "type": "url",
                                "url": "https://example.com/weather-search",
                            }
                        ],
                        "type": "search",
                    },
                    "status": "completed",
                    "type": "web_search_call",
                },
                "output_index": 0,
                "sequence_number": 6,
                "type": "response.output_item.done",
            },
            {
                "type": "response.output_text.delta",
                "delta": "Kyiv weather is mild.",
                "item_id": "msg_id",
                "output_index": 0,
                "content_index": 0,
            },
            {
                "type": "response.output_text.annotation.added",
                "annotation": {
                    "type": "url_citation",
                    "start_index": 0,
                    "end_index": 10,
                    "title": "Kyiv weather source",
                    "url": "https://example.com/weather/kyiv",
                },
                "annotation_index": 0,
                "item_id": "msg_id",
                "output_index": 0,
                "content_index": 0,
            },
            {
                "type": "response.completed",
                "response": completed_response.model_dump(),
            },
        ]
        return httpx.Response(
            status_code=200,
            content=_to_sse_content(events),
            headers={"Content-Type": "text/event-stream"},
        )

    respx.post("http://localhost:5001/openai/v1/responses").mock(
        side_effect=check_request
    )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the weather in Kyiv now? Include source links.",
                }
            ],
            "web_search_options": {
                "user_location": {"city": "London", "type": "approximate"}
            },
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/v1/responses",
        },
    )

    assert response.status_code == 200
    chunks, saw_done = _read_stream_chunks(response)
    assert saw_done

    assert chunks[1]["choices"][0]["delta"]["custom_content"] == {
        "stages": [{"index": 0, "name": "Web Search"}]
    }
    assert chunks[2]["choices"][0]["delta"]["custom_content"]["stages"] == [
        {
            "index": 0,
            "content": (
                "Search\n\nQueries:\n- Weather in Kyiv\n\nSources:\n"
                "- https://example.com/weather-search"
            ),
        }
    ]
    assert chunks[3]["choices"][0]["delta"]["custom_content"] == {
        "stages": [{"index": 0, "status": "completed"}]
    }
    assert chunks[4]["choices"][0]["delta"]["role"] is None
    assert (
        chunks[4]["choices"][0]["delta"]["content"] == "Kyiv weather is mild."
    )
    assert chunks[5]["choices"][0]["delta"]["annotations"] == [
        {
            "type": "url_citation",
            "url_citation": {
                "start_index": 0,
                "end_index": 10,
                "title": "Kyiv weather source",
                "url": "https://example.com/weather/kyiv",
            },
        }
    ]
    assert chunks[6]["choices"][0]["delta"]["custom_content"] == {
        "attachments": [
            {
                "type": "text/markdown",
                "title": "Kyiv weather source",
                "url": "https://example.com/weather/kyiv",
            }
        ]
    }
    assert chunks[7]["choices"][0]["finish_reason"] == "stop"


def test_convert_tools_web_search():
    tools = convert_tools(
        [{"type": "static_function", "static_function": {"name": "web_search"}}]
    )
    assert tools == [{"type": "web_search"}]


def test_convert_tools_web_search_with_extra_fields():
    tools = convert_tools(
        [
            {
                "type": "static_function",
                "static_function": {
                    "name": "web_search",
                    "configuration": {"search_context_size": "high"},
                },
            }
        ]
    )
    assert tools == [{"type": "web_search", "search_context_size": "high"}]


def test_convert_tools_web_search_with_unsupported_name():
    with pytest.raises(RequestValidationError):
        convert_tools(
            [
                {
                    "type": "static_function",
                    "static_function": {"name": "web_search_preview"},
                }
            ]
        )


def test_convert_tools_web_search_without_static_function_payload():
    with pytest.raises(RequestValidationError):
        convert_tools([{"type": "static_function"}])


def test_convert_response_with_web_search_call():
    response = Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            _response_web_search(item_id="ws_id", query="weather Kyiv"),
            _response_output_message("The weather in Kyiv is sunny."),
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    chat_completion = convert_response(response)
    assert (
        chat_completion.choices[0].message.content
        == "The weather in Kyiv is sunny."
    )
    assert chat_completion.choices[0].message.tool_calls is None
    message_dump = chat_completion.choices[0].message.model_dump()
    assert message_dump["custom_content"] == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": "Search\n\nQueries:\n- weather Kyiv",
            }
        ],
        "state": {
            "responses_output": [
                {
                    "id": "ws_id",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["weather Kyiv"],
                        "query": "weather Kyiv",
                        "sources": None,
                    },
                }
            ]
        },
    }
    assert chat_completion.choices[0].finish_reason == "stop"


def test_convert_response_with_multiple_web_search_calls():
    response = Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            _response_web_search(item_id="ws_id_1", query="weather Kyiv"),
            _response_web_search(item_id="ws_id_2", query="news Kyiv"),
            _response_output_message("The weather in Kyiv is sunny."),
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    chat_completion = convert_response(response)
    assert (
        chat_completion.choices[0].message.content
        == "The weather in Kyiv is sunny."
    )
    assert chat_completion.choices[0].message.tool_calls is None
    message_dump = chat_completion.choices[0].message.model_dump()
    assert message_dump["custom_content"] == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": "Search\n\nQueries:\n- weather Kyiv",
            },
            {
                "name": "Web Search #2",
                "status": "completed",
                "content": "Search\n\nQueries:\n- news Kyiv",
            },
        ],
        "state": {
            "responses_output": [
                {
                    "id": "ws_id_1",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["weather Kyiv"],
                        "query": "weather Kyiv",
                        "sources": None,
                    },
                },
                {
                    "id": "ws_id_2",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["news Kyiv"],
                        "query": "news Kyiv",
                        "sources": None,
                    },
                },
            ]
        },
    }


def test_convert_response_with_web_search_multiple_queries():
    response = Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            ResponseFunctionWebSearch(
                id="ws_id",
                type="web_search_call",
                status="completed",
                action=ActionSearch(
                    type="search",
                    # Kept only for SDK validation compatibility.
                    query="legacy query",
                    queries=["weather Kyiv", "news Kyiv"],
                    sources=[
                        ActionSearchSource(
                            type="url",
                            url="https://example.com/weather-kyiv",
                        ),
                        ActionSearchSource(
                            type="url",
                            url="https://example.com/news-kyiv",
                        ),
                    ],
                ),
            ),
            _response_output_message("The weather in Kyiv is sunny."),
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    chat_completion = convert_response(response)
    message_dump = chat_completion.choices[0].message.model_dump()
    assert message_dump["custom_content"] == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": (
                    "Search\n\nQueries:\n- weather Kyiv\n- news Kyiv\n\nSources:\n"
                    "- https://example.com/weather-kyiv\n"
                    "- https://example.com/news-kyiv"
                ),
            }
        ],
        "state": {
            "responses_output": [
                {
                    "id": "ws_id",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": ["weather Kyiv", "news Kyiv"],
                        "query": "legacy query",
                        "sources": [
                            {
                                "type": "url",
                                "url": "https://example.com/weather-kyiv",
                            },
                            {
                                "type": "url",
                                "url": "https://example.com/news-kyiv",
                            },
                        ],
                    },
                }
            ]
        },
    }


def test_convert_response_with_web_search_sources_only():
    response = Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            ResponseFunctionWebSearch(
                id="ws_id",
                type="web_search_call",
                status="completed",
                action=ActionSearch(
                    type="search",
                    # Kept only for SDK validation compatibility.
                    query="legacy query",
                    queries=[],
                    sources=[
                        ActionSearchSource(
                            type="url",
                            url="https://example.com/weather-1",
                        ),
                        ActionSearchSource(
                            type="url",
                            url="https://example.com/weather-2",
                        ),
                    ],
                ),
            ),
            _response_output_message("The weather in Kyiv is sunny."),
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )

    chat_completion = convert_response(response)
    message_dump = chat_completion.choices[0].message.model_dump()
    assert message_dump["custom_content"] == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": (
                    "Search\n\nSources:\n- https://example.com/weather-1\n"
                    "- https://example.com/weather-2"
                ),
            }
        ],
        "state": {
            "responses_output": [
                {
                    "id": "ws_id",
                    "type": "web_search_call",
                    "status": "completed",
                    "action": {
                        "type": "search",
                        "queries": [],
                        "query": "legacy query",
                        "sources": [
                            {
                                "type": "url",
                                "url": "https://example.com/weather-1",
                            },
                            {
                                "type": "url",
                                "url": "https://example.com/weather-2",
                            },
                        ],
                    },
                }
            ]
        },
    }
