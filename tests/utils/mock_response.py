import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self, TypedDict, runtime_checkable

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


class SSEEvent(TypedDict):
    event: str | None
    data: Any


@dataclass
class ResponsesAPIEventStream:
    events: list[SSEEvent]

    @classmethod
    def parse(cls, content: bytes) -> "ResponsesAPIEventStream":
        events: list[SSEEvent] = []
        lines = [
            ln for line in content.decode().splitlines() if (ln := line.strip())
        ]
        while lines:
            line = lines.pop()
            if not line.startswith("data: "):
                raise ValueError(
                    f"Invalid data entry in the SSE stream: {line[:100]}"
                )

            data_str = line.removeprefix("data: ")
            data = _coerce_timestamps_to_float(json.loads(data_str))

            event = None
            if lines and lines[-1].startswith("event:"):
                event = lines.pop().removeprefix("event: ")
            events.append({"event": event, "data": data})
        events.reverse()
        return cls(events=events)

    @property
    def json(self) -> list[SSEEvent]:
        return self.events

    @property
    def text(self) -> str:
        blocks: list[str] = []
        for entry in self.events:
            block = ""
            if event := entry["event"]:
                block += f"event: {event}\n"
            data = json.dumps(entry["data"])
            block += f"data: {data}"
            blocks.append(block)
        return "\n\n".join(blocks)


@dataclass
class ResponsesAPIResponse:
    response: dict

    @classmethod
    def parse(cls, content: bytes) -> "ResponsesAPIResponse":
        try:
            streaming_resp = ResponsesAPIEventStream.parse(content)
            response = streaming_resp.events[-1]["data"]["response"]
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
        return json.dumps(self.response)


@dataclass
class ResponsesAPIMockResponse:
    source: Path | bytes

    def parse(self, stream: bool):
        if isinstance(self.source, Path):
            content = self.source.read_bytes()
        else:
            content = self.source

        if stream:
            return ResponsesAPIEventStream.parse(content)
        else:
            return ResponsesAPIResponse.parse(content)


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
