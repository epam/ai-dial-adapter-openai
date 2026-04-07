import inspect
import json
from pathlib import Path
from typing import Awaitable, Callable, assert_never

import httpx
import respx
from pydantic import BaseModel

from tests.utils.mock_response import MockResponse, ResponsesAPIMockResponse

_Response = httpx.Response | dict | str | bytes | BaseModel | MockResponse
_AResponse = Awaitable[_Response] | _Response
_RequestHandler = Callable[[httpx.Request], _AResponse] | _AResponse


class MockServer:
    def post(self, url: str):
        def decorator(handler: _RequestHandler):
            async def mock_handler(request: httpx.Request) -> httpx.Response:
                resp = handler(request) if callable(handler) else handler

                if inspect.isawaitable(resp):
                    resp = await resp

                if isinstance(resp, MockResponse):
                    stream = await _is_streaming(request)
                    content = resp.parse(stream).text
                    content_type = (
                        "text/event-stream" if stream else "application/json"
                    )
                    return httpx.Response(
                        status_code=200,
                        content=content,
                        headers={"content-type": content_type},
                    )

                if isinstance(resp, str | bytes):
                    return httpx.Response(status_code=200, content=resp)

                if isinstance(resp, dict):
                    return httpx.Response(status_code=200, json=resp)

                if isinstance(resp, BaseModel):
                    return httpx.Response(
                        status_code=200, json=resp.model_dump()
                    )

                if isinstance(resp, httpx.Response):
                    return resp

                assert_never(resp)

            respx.post(url).mock(side_effect=mock_handler)

        return decorator

    @classmethod
    def mock_responses_api_response(
        cls, filepath: str
    ) -> ResponsesAPIMockResponse:
        path = (
            Path(__file__).parent.parent
            / "unit_tests"
            / "mock_responses"
            / "responses_api"
            / filepath
        )
        assert path.exists()
        return ResponsesAPIMockResponse(path)


async def _is_streaming(request: httpx.Request) -> bool:
    request_body = json.loads(await request.aread())
    return request_body.get("stream")
