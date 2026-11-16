import json
import logging
from collections.abc import Mapping
from http import HTTPStatus
from typing import Any

import httpx
from fastapi import Request as FastAPIRequest
from fastapi.responses import Response as FastAPIResponse
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from openai._models import FinalRequestOptions

from aidial_adapter_openai.configuration.app_config import DeploymentAPIType
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.decisions.attachments import (
    DECISIONS_ATTACHMENT_RULES,
)
from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.responses.request import (
    download_dial_urls_in_request,
)
from aidial_adapter_openai.utils.caching import (
    build_cache_headers,
    get_decisions_breakpoint_path,
)
from aidial_adapter_openai.utils.client import get_client
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    bad_upstream_endpoint,
    decisions_parser,
    parse_body,
)
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)

# Matches the Decisions `prompt_cache_options.ttl` default of "30m",
# the only documented value.
_DECISIONS_TTL_SEC = 30 * 60

# The upstream body is relayed decoded,
# so the upstream's content framing no longer describes it.
_DROPPED_HEADERS = ("content-length", "content-encoding")


def _read_input_tokens(content: bytes) -> int | None:
    try:
        body = json.loads(content)
    except ValueError:
        return None

    usage = body.get("usage") if isinstance(body, dict) else None
    tokens = usage.get("input_tokens") if isinstance(usage, dict) else None
    if isinstance(tokens, int) and not isinstance(tokens, bool):
        return tokens
    return None


async def _get_cache_headers(
    request: FastAPIRequest, request_body: Mapping[str, Any], content: bytes
) -> dict[str, str] | None:
    input_tokens = _read_input_tokens(content)
    if input_tokens is None:
        return None

    # The usage reported by the upstream response stands in
    # for the request tokens
    async def get_tokens() -> int:
        return input_tokens

    return await build_cache_headers(
        request_headers=request.headers,
        breakpoint_path=get_decisions_breakpoint_path(request_body),
        get_request_tokens=get_tokens,
        ttl_sec=_DECISIONS_TTL_SEC,
    )


async def decisions_create(request: FastAPIRequest) -> FastAPIResponse:
    request_body = await parse_body(request)
    upstream_endpoint = get_upstream_endpoint(request.headers)
    upstream_extra_headers = get_upstream_extra_headers(request.headers)

    endpoint = decisions_parser.parse(upstream_endpoint)
    # Checked before the client is built, since the dated Azure client
    # would fail first for the lack of api-version
    if isinstance(endpoint, AzureOpenAIEndpoint):
        raise bad_upstream_endpoint(
            "Only v1 API upstream endpoints are supported for Decisions API deployments"
        )

    client = await get_client(
        request=request,
        deployment_id=request_body.get_str("model"),
        deployment=DeploymentAPIType(
            deployment_type=D.DECISIONS_API, endpoint=endpoint
        ),
        app_config=get_request_app_config(request),
        extra_headers=upstream_extra_headers,
        api_version=None,
    )
    if not isinstance(
        client, AsyncAzureOpenAI | AsyncOpenAI | AsyncBedrockOpenAI
    ):
        raise ValueError(
            f"Unexpected client for the deployment backed by Decisions API - {type(client)}"
        )

    raw_body = request_body.unwrap_dict()
    request_modified = await download_dial_urls_in_request(
        create_file_storage(request.headers),
        raw_body,
        rules=DECISIONS_ATTACHMENT_RULES,
    )

    # The query string is not forwarded, api-version included
    response = await client.request(
        cast_to=httpx.Response,
        options=FinalRequestOptions.construct(
            method="post",
            url="/decisions",
            json_data=raw_body if request_modified else await request.body(),
        ),
        stream=False,
    )
    content = await response.aread()

    headers = {
        k: v for k, v in response.headers.items() if k not in _DROPPED_HEADERS
    }
    if response.status_code == HTTPStatus.OK:
        headers |= await _get_cache_headers(request, raw_body, content) or {}

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"response: {content.decode(errors='replace')}")

    return FastAPIResponse(
        content=content, status_code=response.status_code, headers=headers
    )
