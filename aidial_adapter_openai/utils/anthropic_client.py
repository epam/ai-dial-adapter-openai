from functools import cache

import anthropic
import httpx
from anthropic import AsyncAnthropicFoundry

from aidial_adapter_openai.utils.auth import get_azure_access_token


@cache
def get_anthropic_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=_get_default_anthropic_timeout())


async def get_anthropic_client(
    api_key: str | None, base_url: str
) -> AsyncAnthropicFoundry:
    token_provider = get_azure_access_token if api_key is None else None
    return AsyncAnthropicFoundry(
        api_key=api_key,
        base_url=base_url,
        azure_ad_token_provider=token_provider,
        http_client=get_anthropic_httpx_client(),
        max_retries=0,
    )


def _get_default_anthropic_timeout() -> httpx.Timeout:
    # Providing a timeout marginally different from the default Anthropic timeout
    # in order to disable the check that throws an error when
    # stream=False & max_tokens>=128K/6:
    # https://github.com/anthropics/anthropic-sdk-python/blob/f5bdf5137cc3da4d3663aedb8c63d54652981c3b/src/anthropic/resources/beta/messages/messages.py#L2175-L2176

    timeout = anthropic._constants.DEFAULT_TIMEOUT.as_dict()
    timeout["connect"] *= 1.0001  # type: ignore
    return httpx.Timeout(**timeout)
