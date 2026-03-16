import gzip
import json
from typing import Callable, Mapping

import httpx
import openai
import pytest
import respx
from openai import AsyncOpenAI
from openai.types.responses import Response
from openai.types.responses.response_create_params import (
    ResponseCreateParamsNonStreaming,
)

from tests.conftest import OpenAIClientFactory
from tests.utils.mock_server import MockServer

_Headers = Mapping[str, str]
_RequestHandler = Callable[[dict, _Headers], httpx.Response | dict]


class TestResponsesEndpoint:
    UPSTREAM_KEY = "test-upstream-key"
    UPSTREAM_ENDPOINT = "http://localhost:5001/openai/v1/responses"
    UPSTREAM_MODEL = "test-upstream-model-name"

    TEST_RESPONSE = Response(
        id="id",
        created_at=0,
        model=UPSTREAM_MODEL,
        object="response",
        output=[],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    ).model_dump()

    @property
    def test_request(self) -> ResponseCreateParamsNonStreaming:
        return ResponseCreateParamsNonStreaming(
            model=self.UPSTREAM_MODEL, input="Test content"
        )

    @pytest.fixture()
    def client(self, create_openai_client: OpenAIClientFactory):
        return create_openai_client(
            upstream_endpoint=self.UPSTREAM_ENDPOINT,
            upstream_key=self.UPSTREAM_KEY,
        )

    def mock_upstream_response(self):
        def _dec(handler: _RequestHandler):
            @MockServer().post(self.UPSTREAM_ENDPOINT)
            def _responses(request: httpx.Request):
                body = json.loads(request.content)
                return handler(body, request.headers)

        return _dec

    @respx.mock
    async def test_authz(self, client: AsyncOpenAI):
        @self.mock_upstream_response()
        def _handler(body, headers):
            assert headers.get("authorization") == f"Bearer {self.UPSTREAM_KEY}"
            return self.TEST_RESPONSE

        await client.responses.create(**self.test_request)

    @respx.mock
    async def test_model_name(self, client: AsyncOpenAI):
        @self.mock_upstream_response()
        def _handler(body, headers):
            assert body["model"] == self.UPSTREAM_MODEL
            return self.TEST_RESPONSE

        await client.responses.create(**self.test_request)

    @respx.mock
    async def test_proxy_headers_from_upstream(self, client: AsyncOpenAI):
        @self.mock_upstream_response()
        def _handler(body, headers):
            return httpx.Response(
                status_code=200,
                json=self.TEST_RESPONSE,
                headers={"foo": "bar"},
            )

        response = await client.responses.with_raw_response.create(
            **self.test_request
        )

        assert response.status_code == 200
        assert response.headers.get("foo") == "bar"

    @respx.mock
    async def test_gzip_encoding(self, client: AsyncOpenAI):
        @self.mock_upstream_response()
        def _handler(body, headers):
            body_bytes = json.dumps(self.TEST_RESPONSE).encode()
            body = gzip.compress(body_bytes)

            return httpx.Response(
                status_code=200,
                content=body,
                headers={
                    "Content-Length": "1",  # plainly false value
                    "Content-Encoding": "gzip",
                    "Content-Type": "application/json",
                },
            )

        response = await client.responses.with_raw_response.create(
            **self.test_request
        )

        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
        assert response.headers.get("Content-Length") != "1"
        assert response.headers.get("Content-Type") == "application/json"

    @respx.mock
    async def test_too_many_requests_error(self, client: AsyncOpenAI):
        @self.mock_upstream_response()
        def _handler(body, headers):
            return httpx.Response(
                status_code=429,
                content="Too many requests",
                headers={"retry-after": "15"},
            )

        with pytest.raises(openai.RateLimitError) as exc:
            await client.responses.create(**self.test_request)

        error = exc.value
        assert error.status_code == 429
        assert error.message == "Too many requests"
        assert error.response.headers.get("retry-after") == "15"
