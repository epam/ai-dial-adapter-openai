from collections.abc import AsyncIterator

from fastapi import Request
from fastapi.responses import Response as FastAPIResponse
from openai import AsyncStream
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


async def responses(request: Request) -> FastAPIResponse:
    app_config = get_request_app_config(request)
    response = await _responses(request)
    return await create_server_response(
        response,
        emulate_streaming=False,
        sse_stream_format="responses",
        sse_heartbeat_interval=app_config.SSE_HEARTBEAT_INTERVAL,
    )


async def _responses(
    request: Request,
) -> ResponseWithHeaders[dict | AsyncIterator[dict]]:
    request_body = await parse_body(request)

    headers = request.headers
    file_storage = create_file_storage(headers)
    upstream_endpoint = get_upstream_endpoint(headers)
    creds = await get_credentials(headers, azure=True)
    upstream_extra_headers = get_upstream_extra_headers(headers)

    api_version = request.query_params.get("api-version")

    endpoint = responses_parser.parse(upstream_endpoint)
    client = endpoint.get_client(
        {**creds, "api_version": api_version, "headers": upstream_extra_headers}
    )

    request_body = await download_dial_urls_in_request(
        file_storage, request_body
    )

    response: LegacyAPIResponse[
        Response | AsyncStream[ResponseStreamEvent]
    ] = await call_with_extra_body(
        client.responses.with_raw_response.create, request_body
    )

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
