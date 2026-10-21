from collections.abc import AsyncIterator

from aidial_adapter_openai.utils.sse_stream import chunk_sse_events

_EVENTS = [
    b'event: response.created\ndata: {"type":"response.created"}\n\n',
    b'event: response.output_text.delta\ndata: {"delta":"hi"}\n\n',
    b'data: {"type":"response.completed"}\n\n',
]
_STREAM = b"".join(_EVENTS)


async def _rechunk(*chunks: bytes) -> list[bytes]:
    async def _stream() -> AsyncIterator[bytes]:
        for chunk in chunks:
            yield chunk

    return [event async for event in chunk_sse_events(_stream())]


async def test_whole_events_from_any_split():
    """
    Wherever the transport cuts the stream - mid-event, mid-terminator - the
    re-chunked items are the events themselves.
    """
    for cut in range(1, len(_STREAM)):
        assert await _rechunk(_STREAM[:cut], _STREAM[cut:]) == _EVENTS, cut


async def test_crlf_terminators():
    events = [event.replace(b"\n", b"\r\n") for event in _EVENTS]
    stream = b"".join(events)
    chunks = [stream[i : i + 7] for i in range(0, len(stream), 7)]

    assert await _rechunk(*chunks) == events


async def test_trailing_bytes_are_flushed():
    assert await _rechunk(b"data: {}\n\ndata: cut") == [
        b"data: {}\n\n",
        b"data: cut",
    ]


async def test_empty_stream():
    assert await _rechunk() == []
