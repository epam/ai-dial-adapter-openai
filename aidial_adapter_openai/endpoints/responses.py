import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from http import HTTPStatus
from typing import Any, Self, cast

import httpx
from fastapi import Request as FastAPIRequest
from fastapi.responses import Response as FastAPIResponse
from fastapi.responses import StreamingResponse
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from openai._models import FinalRequestOptions
from openai.types.responses.response_create_params import (
    ResponseCreateParamsBase,
)

from aidial_adapter_openai.configuration.app_config import DeploymentAPIType
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.request import (
    get_upstream_endpoint,
)
from aidial_adapter_openai.dial_api.storage import (
    create_file_storage,
)
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)
from aidial_adapter_openai.utils.caching import (
    build_cache_headers,
    get_responses_breakpoint_path,
)
from aidial_adapter_openai.utils.client import get_client
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    parse_body,
    responses_parser,
)
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.sse_stream import chunk_sse_events
from aidial_adapter_openai.utils.streaming import add_sse_bytes_heartbeat
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)

_PATH = "/responses"

# The upstream body is relayed decoded - the SSE framing has to be legible to
# re-chunk it - so the upstream's content framing no longer describes it.
_DROPPED_HEADERS = ("content-length", "content-encoding")


def _to_text(content: bytes) -> str:
    return content.decode(errors="replace")


async def _iter_sse_events(response: httpx.Response) -> AsyncIterator[bytes]:
    try:
        async for event in chunk_sse_events(response.aiter_bytes()):
            yield event
    finally:
        await response.aclose()


async def _log_events(stream: AsyncIterator[bytes]) -> AsyncIterator[bytes]:
    async for event in stream:
        logger.debug(f"event: {_to_text(event)}")
        yield event


@dataclass
class _Response:
    content: AsyncIterator[bytes] | bytes
    status_code: int
    headers: dict[str, str]

    def add_debug_logs(self) -> Self:
        if not logger.isEnabledFor(logging.DEBUG):
            return self

        if isinstance(self.content, bytes):
            logger.debug(f"response: {_to_text(self.content)}")
            return self
        else:
            return replace(self, content=_log_events(self.content))

    def add_heartbeat(self, interval: float | None) -> Self:
        if isinstance(self.content, bytes):
            return self
        else:
            return replace(
                self, content=add_sse_bytes_heartbeat(self.content, interval)
            )

    def to_fastapi_response(self) -> FastAPIResponse:
        if isinstance(self.content, bytes):
            return FastAPIResponse(
                content=self.content,
                status_code=self.status_code,
                headers=self.headers,
            )
        else:
            return StreamingResponse(
                content=self.content,
                status_code=self.status_code,
                headers=self.headers,
            )


async def _prepare_request_body(
    request: FastAPIRequest, request_body: dict[str, Any]
) -> dict[str, Any] | bytes:
    file_storage = create_file_storage(request.headers)
    request_modified = await download_dial_urls_in_request(
        file_storage, request_body
    )

    return request_body if request_modified else await request.body()


@dataclass
class _ResponsesContext:
    request: FastAPIRequest
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI
    query_params: dict[str, str]

    @classmethod
    async def from_request(
        cls, request: FastAPIRequest, *, model: str | None = None
    ) -> Self:
        headers = request.headers
        upstream_endpoint = get_upstream_endpoint(headers)
        upstream_extra_headers = get_upstream_extra_headers(headers)
        query_params = dict(request.query_params)
        app_config = get_request_app_config(request)
        endpoint = responses_parser.parse(upstream_endpoint)
        deployment = DeploymentAPIType(
            deployment_type=D.RESPONSES_API, endpoint=endpoint
        )
        api_version = query_params.pop("api-version", None)

        client = await get_client(
            request=request,
            deployment_id=model,
            deployment=deployment,
            app_config=app_config,
            extra_headers=upstream_extra_headers,
            api_version=api_version,
        )
        if not isinstance(
            client, AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI
        ):
            raise ValueError(
                f"Unexpected client for the deployment backed by Responses API - {type(client)}"
            )

        return cls(request=request, client=client, query_params=query_params)

    async def proxy(
        self,
        path: str,
        *,
        body: dict | bytes | None = None,
        stream: bool = False,
        dial_caching_headers: dict[str, str] | None = None,
    ) -> FastAPIResponse:
        response = await self.client.request(
            cast_to=httpx.Response,
            options=FinalRequestOptions.construct(
                method=self.request.method.lower(),
                url=path,
                json_data=body,
                params=self.query_params,
            ),
            stream=stream,
        )

        app_config = get_request_app_config(self.request)
        headers = {
            k: v
            for k, v in response.headers.items()
            if k not in _DROPPED_HEADERS
        }
        if response.status_code == HTTPStatus.OK:
            headers |= dial_caching_headers or {}

        content = (
            _iter_sse_events(response) if stream else await response.aread()
        )

        return (
            _Response(
                content=content,
                status_code=response.status_code,
                headers=headers,
            )
            .add_debug_logs()
            .add_heartbeat(app_config.SSE_HEARTBEAT_INTERVAL)
            .to_fastapi_response()
        )


async def responses_create(request: FastAPIRequest) -> FastAPIResponse:
    request_body = await parse_body(request)
    context = await _ResponsesContext.from_request(
        request, model=request_body.get("model")
    )

    return await context.proxy(
        _PATH,
        body=await _prepare_request_body(request, request_body),
        stream=request_body.get("stream") is True,
        dial_caching_headers=await build_cache_headers(
            request_headers=request.headers,
            breakpoint_path=get_responses_breakpoint_path(
                cast(ResponseCreateParamsBase, request_body)
            ),
        ),
    )


async def responses_retrieve(
    responses_id: str, request: FastAPIRequest
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    return await context.proxy(
        f"{_PATH}/{responses_id}",
        stream=context.query_params.get("stream") == "true",
    )


async def responses_cancel(
    responses_id: str, request: FastAPIRequest
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    return await context.proxy(f"{_PATH}/{responses_id}/cancel")


async def responses_delete(
    responses_id: str, request: FastAPIRequest
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    return await context.proxy(f"{_PATH}/{responses_id}")


async def responses_input_tokens(request: FastAPIRequest) -> FastAPIResponse:
    request_body = await parse_body(request)
    context = await _ResponsesContext.from_request(
        request, model=request_body.get("model")
    )

    return await context.proxy(
        f"{_PATH}/input_tokens",
        body=await _prepare_request_body(request, request_body),
    )
