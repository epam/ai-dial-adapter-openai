from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Self, overload

from fastapi import Request
from fastapi.responses import Response as FastAPIResponse
from openai import AsyncAzureOpenAI, AsyncOpenAI, AsyncStream
from openai._legacy_response import LegacyAPIResponse
from openai.types.responses import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent

from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.dial_api.storage import (
    create_file_storage,
)
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import (
    parse_body,
    responses_parser,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.streaming import (
    ResponseWithHeaders,
    create_server_response,
    debug_print,
    map_stream,
)
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


@dataclass
class _ResponsesContext:
    client: AsyncAzureOpenAI | AsyncOpenAI
    query_params: dict[str, str]

    @classmethod
    async def from_request(cls, request: Request) -> Self:
        headers = request.headers
        upstream_endpoint = get_upstream_endpoint(headers)
        creds = await get_credentials(headers, azure=True)
        upstream_extra_headers = get_upstream_extra_headers(headers)

        query_params = dict(request.query_params)
        api_version = query_params.pop("api-version", None)

        endpoint = responses_parser.parse(upstream_endpoint)
        client = endpoint.get_client(
            {
                **creds,
                "api_version": api_version,
                "headers": upstream_extra_headers,
            }
        )

        return cls(client=client, query_params=query_params)


async def post_responses(request: Request) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)

    request_body = await parse_body(request)
    file_storage = create_file_storage(request.headers)
    request_body = await download_dial_urls_in_request(
        file_storage, request_body
    )

    response: LegacyAPIResponse[
        Response | AsyncStream[ResponseStreamEvent]
    ] = await call_with_extra_body(
        context.client.responses.with_raw_response.create, request_body
    )

    response_with_headers = _to_response_with_headers(response)
    return await _to_fast_api_response(request, response_with_headers)


async def get_responses(responses_id: str, request: Request) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    stream: bool = context.query_params.get("stream") == "true"
    response: LegacyAPIResponse[
        Response | AsyncStream[ResponseStreamEvent]
    ] = await context.client.responses.with_raw_response.retrieve(
        responses_id,
        stream=stream,
        extra_query=context.query_params,
    )
    response_with_headers = _to_response_with_headers(response)
    return await _to_fast_api_response(request, response_with_headers)


async def post_responses_cancel(
    responses_id: str, request: Request
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    response = await context.client.responses.with_raw_response.cancel(
        responses_id,
        extra_query=context.query_params,
    )
    response_with_headers = _to_response_with_headers(response)
    return await _to_fast_api_response(request, response_with_headers)


async def delete_responses(
    responses_id: str, request: Request
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    response = await context.client.responses.with_raw_response.delete(
        responses_id,
        extra_query=context.query_params,
    )
    return FastAPIResponse(
        content=response.http_response.content,
        headers=response.http_response.headers,
        status_code=response.http_response.status_code,
    )


async def _to_fast_api_response(
    request: Request, response: ResponseWithHeaders[dict | AsyncIterator[dict]]
) -> FastAPIResponse:
    app_config = get_request_app_config(request)
    return await create_server_response(
        response,
        emulate_streaming=False,
        sse_stream_format="responses",
        sse_heartbeat_interval=app_config.SSE_HEARTBEAT_INTERVAL,
    )


@overload
def _to_response_with_headers(
    response: LegacyAPIResponse[Response],
) -> ResponseWithHeaders[dict]: ...


@overload
def _to_response_with_headers(
    response: LegacyAPIResponse[AsyncStream[ResponseStreamEvent]],
) -> ResponseWithHeaders[AsyncIterator[dict]]: ...


def _to_response_with_headers(
    response: LegacyAPIResponse[Response | AsyncStream[ResponseStreamEvent]],
) -> ResponseWithHeaders[dict | AsyncIterator[dict]]:
    response_headers = response.http_response.headers

    # Reformatting of the chunks may invalidate content length.
    # We don't recompress the response, therefore,
    # the content encoding may invalidate too.
    for header in ("content-length", "content-encoding"):
        if header in response_headers:
            del response_headers[header]

    parsed_response = response.parse()

    if isinstance(parsed_response, AsyncStream):
        body = map_stream(_to_dict, parsed_response)
    else:
        body = _to_dict(parsed_response)

    return ResponseWithHeaders(headers=dict(response_headers), body=body)


def _to_dict(obj: ResponseStreamEvent | Response) -> dict:
    ret = obj.to_dict()
    title = "response" if isinstance(obj, Response) else f"event[{obj.type}]"
    debug_print(title, ret)
    return ret
