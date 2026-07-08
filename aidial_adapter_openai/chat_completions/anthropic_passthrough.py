import fastapi
from aidial_adapter_anthropic.passthrough import mount_anthropic_api
from anthropic import AsyncAnthropicFoundry

from aidial_adapter_openai.configuration.app_config import (
    Vendor,
)
from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import (
    anthropic_messages_parser,
    bad_upstream_endpoint,
)


def _strip_unsupported_features(
    client: AsyncAnthropicFoundry, features: list[str]
) -> list[str]:
    _unsupported_flags_by_azure = {"advisor-tool-2026-03-01"}
    return [f for f in features if f not in _unsupported_flags_by_azure]


async def _get_anthropic_client(
    request: fastapi.Request,
) -> AsyncAnthropicFoundry:
    headers = request.headers
    upstream_endpoint = get_upstream_endpoint(headers)

    endpoint = anthropic_messages_parser.try_parse(upstream_endpoint)
    if endpoint is None:
        raise bad_upstream_endpoint(
            "Expected Anthropic API /v1/messages endpoint."
        )

    creds = await get_credentials(headers, vendor=Vendor.AZURE)
    return endpoint.get_client({**creds})


def mount_anthropic_passthrough(app: fastapi.FastAPI, path: str):
    mount_anthropic_api(
        app,
        _get_anthropic_client,
        path=path,
        on_anthropic_beta_header=_strip_unsupported_features,
    )
