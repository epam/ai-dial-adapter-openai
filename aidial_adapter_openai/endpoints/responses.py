from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import Response as FastAPIResponse
from openai import AsyncStream
from openai._legacy_response import LegacyAPIResponse
from openai.types.responses import Response
from openai.types.responses.response_stream_event import ResponseStreamEvent

from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import (
    parse_body,
    responses_parser,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.streaming import (
    create_server_response,
    debug_print,
    map_stream,
)


async def responses(request: Request) -> FastAPIResponse:
    response = await _responses(request)
    return await create_server_response(
        response=response, emulate_streaming=False
    )


async def _responses(
    request: Request,
) -> dict | AsyncIterator[dict]:
    request_body = await parse_body(request)

    upstream_endpoint = get_upstream_endpoint(request.headers)
    creds = await get_credentials(request.headers)
    api_version = request.query_params.get("api-version")

    endpoint = responses_parser.parse(upstream_endpoint)
    client = endpoint.get_client({**creds, "api_version": api_version})

    response: LegacyAPIResponse[
        Response | AsyncStream[ResponseStreamEvent]
    ] = await call_with_extra_body(
        client.responses.with_raw_response.create, request_body
    )

    response_headers = response.http_response.headers

    # Reformatting of the chunk may invalidate content length.
    # We don't recompress the response, therefore content encoding invalidates too.
    for header in ["Content-Length", "Content-Encoding"]:
        if header in response_headers:
            del response_headers[header]

    parsed_response = response.parse()

    if isinstance(parsed_response, AsyncStream):
        return map_stream(_to_dict, parsed_response)
    else:
        return _to_dict(parsed_response)


def _to_dict(obj: ResponseStreamEvent | Response) -> dict:
    ret = obj.to_dict()
    title = "response" if isinstance(obj, Response) else "event"
    debug_print(title, ret)
    return ret
