import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, List, Self, assert_never, runtime_checkable

from typing_extensions import Protocol


class MockResponsePayload(Protocol):
    @classmethod
    def parse(cls, content: bytes) -> Self: ...

    @property
    def text(self) -> str: ...

    @property
    def json(self) -> Any: ...


@runtime_checkable
class MockResponse(Protocol):
    def parse(self, stream: bool) -> MockResponsePayload: ...


@dataclass
class SSEEvent:
    event: str | None
    data: Any


@dataclass
class SSEComment:
    message: str


@dataclass
class ResponsesAPIEventStream:
    entries: list[SSEEvent | SSEComment]

    @classmethod
    def parse(cls, content: bytes) -> "ResponsesAPIEventStream":
        entries: list[SSEEvent | SSEComment] = []
        lines = [
            ln for line in content.decode().splitlines() if (ln := line.strip())
        ]

        while lines:
            line = lines.pop()
            if line.startswith(":"):
                entries.append(SSEComment(line.removeprefix(":")))
                continue

            if not line.startswith("data: "):
                raise ValueError(
                    f"Invalid data entry in the SSE stream: {line[:100]}"
                )

            data_str = line.removeprefix("data: ")
            data = _coerce_timestamps_to_float(json.loads(data_str))

            event = None
            if lines and lines[-1].startswith("event:"):
                event = lines.pop().removeprefix("event: ")
            entries.append(SSEEvent(event, data))

        entries.reverse()
        return cls(entries=entries)

    @property
    def json(self) -> list[dict]:
        return [asdict(e) for e in self.entries]

    @property
    def text(self) -> str:
        blocks: list[str] = []
        for entry in self.entries:
            match entry:
                case SSEComment(message):
                    block = f":{message}"
                case SSEEvent(event, data):
                    block = ""
                    if event:
                        block += f"event: {event}\n"
                    block += f"data: {_compact_json(data)}"
                case _:
                    assert_never(entry)
            blocks.append(block + "\n\n")
        return "".join(blocks)

    def get_last_sse_event(self) -> SSEEvent:
        for e in reversed(self.entries):
            if isinstance(e, SSEEvent):
                return e
        raise ValueError("The stream doesn't contain any SSE events.")

    def signature(self) -> List[str]:
        def _get_name(x: SSEEvent | SSEComment) -> str:
            return "event" if isinstance(x, SSEEvent) else f"comment{x.message}"

        return [_get_name(e) for e in self.entries]


@dataclass
class ResponsesAPIResponse:
    response: dict

    @classmethod
    def parse(cls, content: bytes) -> "ResponsesAPIResponse":
        try:
            stream = ResponsesAPIEventStream.parse(content)
            last_event = stream.get_last_sse_event()
            assert last_event.data["type"] == "response.completed"
            response = last_event.data["response"]
            return cls(response=response)
        except Exception:
            pass

        try:
            response = json.loads(content)
            return cls(response=response)
        except Exception:
            pass

        raise ValueError(
            f"The response is neither a valid JSON, nor a valid SSE stream: {content[:100]}"
        )

    @property
    def json(self) -> dict:
        return self.response

    @property
    def text(self) -> str:
        return _compact_json(self.response)


@dataclass
class ResponsesAPIMockResponse:
    source: Path | bytes

    @property
    def content(self) -> bytes:
        if isinstance(self.source, Path):
            return self.source.read_bytes()
        else:
            return self.source

    def parse_stream(self) -> ResponsesAPIEventStream:
        return ResponsesAPIEventStream.parse(self.content)

    def parse_block(self) -> ResponsesAPIResponse:
        return ResponsesAPIResponse.parse(self.content)

    def parse(self, stream: bool) -> MockResponsePayload:
        return self.parse_stream() if stream else self.parse_block()


def _coerce_timestamps_to_float(obj: Any):
    _rec = _coerce_timestamps_to_float

    if isinstance(obj, list | tuple):
        for x in obj:
            _rec(x)
    elif isinstance(obj, dict):
        for v in obj.values():
            _rec(v)
        for key in ("created_at", "completed_at"):
            if (timestamp := obj.get(key)) and isinstance(timestamp, int):
                obj[key] = float(timestamp)

    return obj


def _compact_json(x: Any) -> str:
    return json.dumps(x, separators=(",", ":"))
