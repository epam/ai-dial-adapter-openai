import json
from typing import Callable

import httpx
import respx
from pydantic import BaseModel

RequestHandler = Callable[[httpx.Request], httpx.Response | dict | BaseModel]


class MockServer:
    def post(self, url: str):
        def _dec(handler: RequestHandler):
            def _handler(request: httpx.Request) -> httpx.Response:
                resp = handler(request)
                if isinstance(resp, dict):
                    return httpx.Response(
                        status_code=200, content=json.dumps(resp)
                    )
                if isinstance(resp, BaseModel):
                    return httpx.Response(
                        status_code=200, content=resp.model_dump_json()
                    )
                return resp

            respx.post(url).mock(side_effect=_handler)

        return _dec
