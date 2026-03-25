import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, runtime_checkable

from typing_extensions import Protocol


@dataclass
class ResponsesAPIMockResponse:
    file: Path

    def get_response(self, *, stream: bool) -> bytes:
        content = self.file.read_bytes()
        if b"data: " in content:
            if stream:
                return content
            else:
                last_data = [
                    line
                    for line in content.decode().splitlines()
                    if line.startswith("data: ")
                ][-1].removeprefix("data: ")
                response_event = json.loads(last_data)["response"]
                return json.dumps(response_event).encode()
        else:
            if stream:
                raise ValueError(
                    f"The mock response is expected to contain SSE stream: {str(self.file)}"
                )
            else:
                return content

    @staticmethod
    def parse(*, stream: bool, content: bytes) -> Any:
        if stream:
            lines = [
                ln
                for line in content.decode().splitlines()
                if (ln := line.strip())
            ]
            events = []
            while lines:
                data_str = lines.pop().removeprefix("data: ")
                try:
                    data = _turn_timestamps_into_floats(json.loads(data_str))
                except Exception:
                    data = data_str

                event = None
                if lines and lines[-1].startswith("event:"):
                    event = lines.pop().removeprefix("event: ")
                events.append({"event": event, "data": data})
            events.reverse()
            return events
        else:
            return json.loads(content)


@runtime_checkable
class MockResponse(Protocol):
    def get_response(self, *, stream: bool) -> bytes: ...

    @staticmethod
    def parse(*, stream: bool, content: bytes) -> Any: ...


def _turn_timestamps_into_floats(obj: Any):
    _rec = _turn_timestamps_into_floats

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
