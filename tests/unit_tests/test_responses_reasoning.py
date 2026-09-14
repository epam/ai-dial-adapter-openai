"""Unit tests for the reasoning content reported by the Responses API."""

import json

import httpx
import respx
from openai.types.responses import (
    Response,
    ResponseCompletedEvent,
    ResponseCreatedEvent,
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
    ResponseReasoningSummaryPartAddedEvent,
    ResponseReasoningSummaryTextDeltaEvent,
    ResponseReasoningSummaryTextDoneEvent,
    ResponseStreamEvent,
    ResponseTextDeltaEvent,
)
from openai.types.responses.response_reasoning_item import Summary
from openai.types.responses.response_reasoning_summary_part_added_event import (
    Part,
)

_UPSTREAM_ENDPOINT = "http://localhost:5001/openai/v1/responses"
_REASONING_ITEM_ID = "rs_id"
_SUMMARIES = ["Part one", "Part two"]

# The summary parts are joined with a blank line in between
_EXPECTED_REASONING_CONTENT = "Part one\n\nPart two"


def _reasoning_item(*summaries: str) -> ResponseReasoningItem:
    return ResponseReasoningItem(
        id=_REASONING_ITEM_ID,
        type="reasoning",
        summary=[Summary(type="summary_text", text=text) for text in summaries],
    )


def _response(
    *,
    status: str,
    with_output: bool,
    reasoning: list[ResponseReasoningItem] | None = None,
) -> Response:
    output = (
        [
            *(reasoning or [_reasoning_item(*_SUMMARIES)]),
            ResponseOutputMessage(
                id="msg_id",
                type="message",
                role="assistant",
                status="completed",
                content=[
                    ResponseOutputText(
                        type="output_text", text="The answer", annotations=[]
                    )
                ],
            ),
        ]
        if with_output
        else []
    )
    return Response(
        id="resp_id",
        created_at=0,
        model="test-model",
        object="response",
        status=status,  # type: ignore
        output=output,  # type: ignore
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _stream_events() -> list[ResponseStreamEvent]:
    events: list[ResponseStreamEvent] = [
        ResponseCreatedEvent(
            type="response.created",
            sequence_number=0,
            response=_response(status="in_progress", with_output=False),
        )
    ]

    for summary_index, summary in enumerate(_SUMMARIES):
        events += [
            ResponseReasoningSummaryPartAddedEvent(
                type="response.reasoning_summary_part.added",
                sequence_number=0,
                item_id=_REASONING_ITEM_ID,
                output_index=0,
                summary_index=summary_index,
                part=Part(type="summary_text", text=""),
            ),
            # The summary is streamed in two deltas
            *(
                ResponseReasoningSummaryTextDeltaEvent(
                    type="response.reasoning_summary_text.delta",
                    sequence_number=0,
                    item_id=_REASONING_ITEM_ID,
                    output_index=0,
                    summary_index=summary_index,
                    delta=delta,
                )
                for delta in (summary[:3], summary[3:])
            ),
            ResponseReasoningSummaryTextDoneEvent(
                type="response.reasoning_summary_text.done",
                sequence_number=0,
                item_id=_REASONING_ITEM_ID,
                output_index=0,
                summary_index=summary_index,
                text=summary,
            ),
        ]

    return events + [
        ResponseTextDeltaEvent(
            type="response.output_text.delta",
            sequence_number=0,
            item_id="msg_id",
            output_index=1,
            content_index=0,
            logprobs=[],
            delta="The answer",
        ),
        ResponseCompletedEvent(
            type="response.completed",
            sequence_number=0,
            response=_response(status="completed", with_output=True),
        ),
    ]


def _mock_upstream(stream: bool) -> None:
    if stream:
        content = "".join(
            f"data: {json.dumps(event.model_dump())}\n\n"
            for event in _stream_events()
        )
        response = httpx.Response(
            status_code=200,
            content=content + "data: [DONE]\n\n",
            headers={"Content-Type": "text/event-stream"},
        )
    else:
        response = httpx.Response(
            status_code=200,
            json=_response(status="completed", with_output=True).model_dump(),
        )

    respx.post(_UPSTREAM_ENDPOINT).mock(return_value=response)


async def _chat_completion(
    test_app: httpx.AsyncClient, stream: bool
) -> httpx.Response:
    _mock_upstream(stream)
    return await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "stream": stream,
            "messages": [{"role": "user", "content": "2+2?"}],
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )


@respx.mock
async def test_reasoning_content_in_response(test_app: httpx.AsyncClient):
    response = await _chat_completion(test_app, stream=False)

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == _EXPECTED_REASONING_CONTENT
    assert message["content"] == "The answer"
    assert message["custom_content"]["stages"] == [
        {"name": "Reasoning", "status": "completed", "content": "Part one"},
        {"name": "Reasoning #2", "status": "completed", "content": "Part two"},
    ]


@respx.mock
async def test_reasoning_content_in_stream(test_app: httpx.AsyncClient):
    response = await _chat_completion(test_app, stream=True)

    assert response.status_code == 200
    deltas = [
        json.loads(line.removeprefix("data: "))["choices"][0]["delta"]
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]

    # The concatenated reasoning content deltas are identical
    # to the reasoning content of the non-streaming response
    assert (
        "".join(delta.get("reasoning_content") or "" for delta in deltas)
        == _EXPECTED_REASONING_CONTENT
    )
    assert "".join(delta.get("content") or "" for delta in deltas) == (
        "The answer"
    )
    assert [
        stage
        for delta in deltas
        for stage in (delta.get("custom_content") or {}).get("stages") or []
    ] == [
        {"index": 0, "name": "Reasoning"},
        {"index": 0, "content": "Par"},
        {"index": 0, "content": "t one"},
        {"index": 0, "status": "completed"},
        {"index": 1, "name": "Reasoning #2"},
        {"index": 1, "content": "Par"},
        {"index": 1, "content": "t two"},
        {"index": 1, "status": "completed"},
    ]


@respx.mock
async def test_reasoning_content_of_multiple_reasoning_items(
    test_app: httpx.AsyncClient,
):
    respx.post(_UPSTREAM_ENDPOINT).mock(
        return_value=httpx.Response(
            status_code=200,
            json=_response(
                status="completed",
                with_output=True,
                reasoning=[_reasoning_item(text) for text in _SUMMARIES],
            ).model_dump(),
        )
    )

    response = await test_app.post(
        "/openai/deployments/adapter-deployment-name/chat/completions?api-version=2023-03-15-preview",
        json={
            "model": "upstream-model-name",
            "messages": [{"role": "user", "content": "2+2?"}],
        },
        headers={
            "X-UPSTREAM-KEY": "test-api-key",
            "X-UPSTREAM-ENDPOINT": _UPSTREAM_ENDPOINT,
        },
    )

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    assert message["reasoning_content"] == _EXPECTED_REASONING_CONTENT
    assert message["custom_content"]["stages"] == [
        {"name": "Reasoning", "status": "completed", "content": "Part one"},
        {"name": "Reasoning #2", "status": "completed", "content": "Part two"},
    ]
