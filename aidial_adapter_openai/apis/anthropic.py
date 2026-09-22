import fastapi
from anthropic import AsyncAnthropic

from aidial_adapter_openai.configuration.app_config import (
    Vendor,
)
from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import (
    anthropic_messages_parser,
    bad_upstream_endpoint,
)


async def get_anthropic_client(
    request: fastapi.Request,
) -> AsyncAnthropic:
    headers = request.headers
    upstream_endpoint = get_upstream_endpoint(headers)

    endpoint = anthropic_messages_parser.try_parse(upstream_endpoint)
    if endpoint is None:
        raise bad_upstream_endpoint(
            "Expected Anthropic API /v1/messages endpoint."
        )

    creds = await get_credentials(
        headers, vendor=Vendor.AZURE, endpoint=None, deployment_id=None
    )
    return endpoint.get_client({**creds})
