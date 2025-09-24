from aidial_adapter_openai.responses.event_handler import EventHandler


def test_event_handler_tool_calls_independence():
    x = EventHandler(id_="id", created_=123, model_="model")
    y = EventHandler(id_="id", created_=123, model_="model")

    chunk = x._tool_call_chunk_open("item_id1", "args1", "name1", "call_id1")
    assert chunk.choices[0].delta.tool_calls[0].index == 0  # type: ignore

    chunk = y._tool_call_chunk_open("item_id2", "args2", "name2", "call_id2")
    assert chunk.choices[0].delta.tool_calls[0].index == 0  # type: ignore
