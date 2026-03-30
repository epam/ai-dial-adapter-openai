import asyncio
import gzip
import inspect
import json
from typing import (
    AsyncIterator,
    Awaitable,
    Generator,
    Mapping,
    Protocol,
)

import httpx
import openai
import pytest
import respx
from openai import AsyncOpenAI
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)

from aidial_adapter_openai.utils.request import get_app_config
from tests.conftest import OpenAIClientFactory
from tests.utils.mock_response import MockResponse, ResponsesAPIMockResponse
from tests.utils.mock_server import MockServer

_Response = httpx.Response | MockResponse | dict


class _RequestHandler(Protocol):
    def __call__(
        self, *, body: dict, headers: Mapping[str, str], stream: bool
    ) -> Awaitable[_Response] | _Response: ...


class TestResponsesEndpoint:
    UPSTREAM_KEY = "test-upstream-key"
    UPSTREAM_ENDPOINT = "http://test-upstream-hostname/openai/v1/responses"
    UPSTREAM_MODEL = "test-upstream-model-name"

    MOCK_RESPONSE = MockServer.mock_responses_api_response("text.txt")

    @pytest.fixture
    def sse_heartbeat_interval_1(self, _app_instance):
        app_config = get_app_config(_app_instance)  # type: ignore
        app_config.SSE_HEARTBEAT_INTERVAL = 1
        yield
        app_config.SSE_HEARTBEAT_INTERVAL = None

    @pytest.fixture()
    def client(self, create_openai_client: OpenAIClientFactory):
        return create_openai_client(
            upstream_endpoint=self.UPSTREAM_ENDPOINT,
            upstream_key=self.UPSTREAM_KEY,
        )

    @pytest.fixture(params=[True, False], ids=["stream", "block"])
    def stream(self, request) -> bool:
        return request.param

    @property
    def test_request(self) -> ResponseCreateParamsBase:
        return ResponseCreateParamsBase(
            model=self.UPSTREAM_MODEL, input="Test content"
        )

    def mock_upstream_response(self, handler: _RequestHandler | MockResponse):
        if isinstance(handler, MockResponse):
            MockServer().post(self.UPSTREAM_ENDPOINT)(handler)
        else:

            @MockServer().post(self.UPSTREAM_ENDPOINT)
            async def _handler(request: httpx.Request):
                body = json.loads(request.content)
                stream = body.get("stream")
                response = handler(
                    body=body,
                    stream=stream,
                    headers=request.headers,
                )
                if inspect.isawaitable(response):
                    return await response
                else:
                    return response

    @respx.mock
    async def test_authz(self, client: AsyncOpenAI, stream: bool):
        @self.mock_upstream_response
        def _handler(headers, **kwargs):
            assert headers.get("authorization") == f"Bearer {self.UPSTREAM_KEY}"
            return self.MOCK_RESPONSE

        await client.responses.create(**self.test_request, stream=stream)

    @respx.mock
    async def test_model_name(self, client: AsyncOpenAI, stream: bool):
        @self.mock_upstream_response
        def _handler(body, **kwargs):
            assert body["model"] == self.UPSTREAM_MODEL
            return self.MOCK_RESPONSE

        await client.responses.create(**self.test_request, stream=stream)

    @respx.mock
    async def test_proxy_headers_from_upstream(
        self, client: AsyncOpenAI, stream: bool
    ):
        @self.mock_upstream_response
        def _handler(stream, **kwargs):
            return httpx.Response(
                status_code=200,
                headers={"foo": "bar"},
                content=self.MOCK_RESPONSE.parse(stream).text,
            )

        response = await client.responses.with_raw_response.create(
            **self.test_request, stream=stream
        )

        assert response.status_code == 200
        assert response.headers.get("foo") == "bar"

    @respx.mock
    async def test_gzip_encoding(self, client: AsyncOpenAI, stream: bool):
        if stream:
            pytest.skip("respx doesn't properly mock compressed responses.")

        content_type = "text/event-stream" if stream else "application/json"

        @self.mock_upstream_response
        def _handler(body, stream, **kwargs):
            gzipped = gzip.compress(
                self.MOCK_RESPONSE.parse(stream).text.encode()
            )

            return httpx.Response(
                status_code=200,
                stream=httpx.ByteStream(gzipped),
                headers={
                    "Content-Length": "1",  # plainly false value
                    "Content-Encoding": "gzip",
                    "Content-Type": content_type,
                },
            )

        response = await client.responses.with_raw_response.create(
            **self.test_request, stream=stream
        )

        assert response.status_code == 200
        assert "Content-Encoding" not in response.headers
        assert response.headers.get("Content-Length") != "1"
        assert response.headers.get("Content-Type") == content_type

    @respx.mock
    async def test_too_many_requests_error(
        self, client: AsyncOpenAI, stream: bool
    ):
        @self.mock_upstream_response
        def _handler(**kwargs):
            return httpx.Response(
                status_code=429,
                content="Too many requests",
                headers={"retry-after": "15"},
            )

        with pytest.raises(openai.RateLimitError) as exc:
            await client.responses.create(**self.test_request, stream=stream)

        error = exc.value
        assert error.status_code == 429
        assert error.message == "Too many requests"
        assert error.response.headers.get("retry-after") == "15"

    @respx.mock
    async def test_responses_streaming(self, client: AsyncOpenAI, stream: bool):
        resp = self.MOCK_RESPONSE
        self.mock_upstream_response(resp)

        response = await client.responses.with_raw_response.create(
            **self.test_request, stream=stream
        )

        actual_content = await response.http_response.aread()

        if stream:
            assert b"event: " in actual_content
            assert b"data: [DONE]" not in actual_content

        expected = resp.parse(stream)
        actual = ResponsesAPIMockResponse(actual_content).parse(stream)
        assert actual.json == expected.json

    @respx.mock
    async def test_extra_request_field(self, client: AsyncOpenAI, stream: bool):
        @self.mock_upstream_response
        def _handler(body, **kwargs):
            assert body.get("extra-field") == "extra-value"
            return self.MOCK_RESPONSE

        await client.responses.create(
            **self.test_request,
            extra_body={"extra-field": "extra-value"},
            stream=stream,
        )  # type: ignore

    @respx.mock
    async def test_sse_heartbeat_interval(
        self, sse_heartbeat_interval_1, client: AsyncOpenAI
    ):
        @self.mock_upstream_response
        async def _handler(body, **kwargs):
            resp = self.MOCK_RESPONSE.parse_stream()

            async def _stream() -> AsyncIterator[bytes]:
                [chunk1, chunk2] = list(_chunk_lines(resp.text, n=2))
                yield chunk1.encode()
                await asyncio.sleep(2)
                yield chunk2.encode()

            return httpx.Response(status_code=200, content=_stream())

        response = await client.responses.with_raw_response.create(
            **self.test_request, stream=True
        )  # type: ignore

        actual_content = await response.http_response.aread()
        actual = ResponsesAPIMockResponse(actual_content).parse_stream()
        assert _is_sublist(
            ["event", "comment ping", "event"], actual.signature()
        )


def _chunk_lines(text: str, *, n: int) -> Generator[str, None, None]:
    lines = text.splitlines(keepends=True)
    size = (len(lines) + n - 1) // n
    for i in range(n):
        block = lines[i * size : (i + 1) * size]
        yield "".join(block)


def _is_sublist(small: list[str], big: list[str]) -> bool:
    return ";".join(small) in ";".join(big)
