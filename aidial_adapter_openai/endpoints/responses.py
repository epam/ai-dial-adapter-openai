from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Self, TypeVar

from fastapi import Request
from fastapi.responses import Response as FastAPIResponse
from openai import (
    AsyncAzureOpenAI,
    AsyncBedrockOpenAI,
    AsyncOpenAI,
    AsyncStream,
)
from openai._legacy_response import LegacyAPIResponse
from openai.types.responses import Response
from openai.types.responses.input_token_count_response import (
    InputTokenCountResponse,
)
from openai.types.responses.response_stream_event import ResponseStreamEvent

from aidial_adapter_openai.configuration.app_config import DeploymentAPIType
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.request import (
    DIAL_OVERRIDE_NAME,
    get_upstream_endpoint,
)
from aidial_adapter_openai.dial_api.storage import (
    create_file_storage,
)
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)
from aidial_adapter_openai.utils.client import get_client
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

_INVALIDATED_RESPONSE_HEADERS = ("content-length", "content-encoding")
_ResponsesPayload = (
    Response | InputTokenCountResponse | AsyncStream[ResponseStreamEvent]
)
_ResponsesPayloadT = TypeVar("_ResponsesPayloadT", bound=_ResponsesPayload)


@dataclass
class _ResponsesContext:
    client: AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI
    query_params: dict[str, str]

    @classmethod
    async def from_request(
        cls, request: Request, *, model: str | None = None
    ) -> Self:
        headers = request.headers
        deployment_id = headers.get(DIAL_OVERRIDE_NAME) or model
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
            deployment_id=deployment_id,
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

        return cls(client=client, query_params=query_params)


async def responses_create(request: Request) -> FastAPIResponse:
    request_body = await parse_body(request)
    context = await _ResponsesContext.from_request(
        request, model=request_body.get("model")
    )

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


async def responses_retrieve(
    responses_id: str, request: Request
) -> FastAPIResponse:
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


async def responses_cancel(
    responses_id: str, request: Request
) -> FastAPIResponse:
    context = await _ResponsesContext.from_request(request)
    response = await context.client.responses.with_raw_response.cancel(
        responses_id,
        extra_query=context.query_params,
    )
    response_with_headers = _to_response_with_headers(response)  # type: ignore
    return await _to_fast_api_response(request, response_with_headers)


async def responses_delete(
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


async def responses_input_tokens(request: Request):
    request_body = await parse_body(request)
    context = await _ResponsesContext.from_request(
        request, model=request_body.get("model")
    )
    file_storage = create_file_storage(request.headers)
    request_body = await download_dial_urls_in_request(
        file_storage, request_body
    )

    response = (
        await context.client.responses.with_raw_response.input_tokens.count(
            **request_body
        )
    )

    response_with_headers = _to_response_with_headers(response)
    return await _to_fast_api_response(request, response_with_headers)


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


def _to_response_with_headers(
    response: LegacyAPIResponse[_ResponsesPayloadT],
) -> ResponseWithHeaders[dict | AsyncIterator[dict]]:
    response_headers = dict(response.http_response.headers)

    # Reformatting or reading content may invalidate content length.
    # We don't recompress the response, therefore,
    # the content encoding may invalidate too.
    for header in _INVALIDATED_RESPONSE_HEADERS:
        for key in list(response_headers):
            if key.lower() == header:
                del response_headers[key]

    parsed_response = response.parse()

    if isinstance(parsed_response, AsyncStream):
        body = map_stream(_to_dict, parsed_response)
    else:
        body = _to_dict(parsed_response)

    return ResponseWithHeaders(
        headers=response_headers,
        body=body,
    )


def _to_dict(
    obj: ResponseStreamEvent | Response | InputTokenCountResponse,
) -> dict:
    ret = obj.to_dict()
    match obj:
        case Response() | InputTokenCountResponse():
            title = "response"
        case _:
            title = f"event[{obj.type}]"
    debug_print(title, ret)
    return ret
