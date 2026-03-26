import fastapi
import httpx
from fastapi.responses import Response, StreamingResponse
from openai import AsyncStream
from openai._legacy_response import LegacyAPIResponse
from openai.types.responses.response_stream_event import ResponseStreamEvent
from starlette.background import BackgroundTask

from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import (
    parse_body,
    responses_parser,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body


async def responses(request: fastapi.Request) -> Response:
    request_body = await parse_body(request)

    upstream_endpoint = get_upstream_endpoint(request.headers)
    creds = await get_credentials(request.headers, azure=True)
    api_version = request.query_params.get("api-version")

    endpoint = responses_parser.parse(upstream_endpoint)
    client = endpoint.get_client({**creds, "api_version": api_version})

    response: (
        LegacyAPIResponse[Response]
        | LegacyAPIResponse[AsyncStream[ResponseStreamEvent]]
    ) = await call_with_extra_body(
        client.responses.with_raw_response.create, request_body
    )

    return _httpx_to_fastapi(response.http_response)


def _httpx_to_fastapi(response: httpx.Response) -> Response:
    excluded_headers = ("content-length", "content-encoding")
    headers = {
        k: v
        for k, v in response.headers.items()
        if k.lower() not in excluded_headers
    }

    if response.is_stream_consumed:
        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=headers,
        )

    return StreamingResponse(
        response.aiter_bytes(),
        status_code=response.status_code,
        headers=headers,
        background=BackgroundTask(response.aclose),
    )
