import json

import httpx
import pytest
import respx
from aidial_sdk.exceptions import RequestValidationError
from openai.types.responses import Response
from openai.types.responses.response_function_web_search import (
    ActionSearch,
    ResponseFunctionWebSearch,
)
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_output_text import ResponseOutputText

from aidial_adapter_openai.responses.converter import (
    convert_response,
    convert_tools,
)


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
            output=[
                ResponseOutputMessage(
                    id="msg_id",
                    type="message",
                    role="assistant",
                    status="completed",
                    content=[
                        ResponseOutputText(
                            type="output_text",
                            text="The weather in Kyiv is sunny.",
                            annotations=[],
                        )
                    ],
                )
            ],
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


def test_convert_response_with_web_search_call():
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
                action=ActionSearch(type="search", query="weather Kyiv"),
            ),
            ResponseOutputMessage(
                id="msg_id",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="The weather in Kyiv is sunny.",
                        annotations=[],
                    )
                ],
            ),
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
    assert chat_completion.choices[0].message.custom_content == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": "type: search\nquery: weather Kyiv",
            }
        ]
    }
    assert chat_completion.choices[0].finish_reason == "stop"


def test_convert_response_with_multiple_web_search_calls():
    response = Response(
        id="id",
        created_at=0,
        model="test-model",
        object="response",
        output=[
            ResponseFunctionWebSearch(
                id="ws_id_1",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="weather Kyiv"),
            ),
            ResponseFunctionWebSearch(
                id="ws_id_2",
                type="web_search_call",
                status="completed",
                action=ActionSearch(type="search", query="news Kyiv"),
            ),
            ResponseOutputMessage(
                id="msg_id",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text",
                        text="The weather in Kyiv is sunny.",
                        annotations=[],
                    )
                ],
            ),
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
    assert chat_completion.choices[0].message.custom_content == {
        "stages": [
            {
                "name": "Web Search",
                "status": "completed",
                "content": "type: search\nquery: weather Kyiv",
            },
            {
                "name": "Web Search #2",
                "status": "completed",
                "content": "type: search\nquery: news Kyiv",
            },
        ]
    }
