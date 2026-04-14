from aidial_adapter_openai.responses.converter import (
    parse_response_url_citation,
)
from aidial_adapter_openai.responses.event_handler import EventHandler


def test_event_handler_tool_calls_independence():
    x = EventHandler(id_="id", created_=123, model_="model")
    y = EventHandler(id_="id", created_=123, model_="model")

    chunk = x._tool_call_chunk_open("item_id1", "args1", "name1", "call_id1")
    assert chunk.choices[0].delta.tool_calls[0].index == 0  # type: ignore

    chunk = y._tool_call_chunk_open("item_id2", "args2", "name2", "call_id2")
    assert chunk.choices[0].delta.tool_calls[0].index == 0  # type: ignore


def test_parse_response_url_citation_parses_valid_annotation():
    annotation = {
        "type": "url_citation",
        "start_index": 0,
        "end_index": 10,
        "title": "Example source",
        "url": "https://example.com",
    }

    parsed = parse_response_url_citation(annotation)

    assert parsed is not None
    assert parsed.type == "url_citation"
    assert parsed.title == "Example source"
    assert parsed.url == "https://example.com"


def test_parse_response_url_citation_ignores_non_url_annotation():
    annotation = {"type": "file_path"}

    parsed = parse_response_url_citation(annotation)

    assert parsed is None


def test_annotation_chunks_ignore_non_url_annotation():
    handler = EventHandler(id_="id", created_=123, model_="model")

    chunks = list(handler._annotation_chunks({"type": "file_path"}))

    assert chunks == []
