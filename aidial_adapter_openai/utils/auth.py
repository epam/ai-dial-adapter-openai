import os
import time
from collections.abc import Mapping
from typing import assert_never

from aidial_sdk.exceptions import HTTPException as DialException
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity.aio import DefaultAzureCredential
from typing_extensions import TypedDict

from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.log_config import logger

EXPIRATION_WINDOW_IN_SEC: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRATION_WINDOW", 10)
)
AZURE_OPEN_AI_SCOPE: str = os.getenv(
    "AZURE_OPEN_AI_SCOPE", "https://cognitiveservices.azure.com/.default"
)


class _AzureTokenProvider:
    _credential: DefaultAzureCredential
    _access_token: AccessToken | None

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        self._access_token = None

    async def aclose(self):
        await self._credential.close()

    async def get_token(self) -> str:
        now = int(time.time())

        if (
            self._access_token is None
            or now + EXPIRATION_WINDOW_IN_SEC > self._access_token.expires_on
        ):
            try:
                self._access_token = await self._credential.get_token(
                    AZURE_OPEN_AI_SCOPE
                )
                logger.debug(
                    f"Obtained new Azure access token, expires on {self._access_token.expires_on}"
                )
            except ClientAuthenticationError as e:
                logger.error(
                    f"Default Azure credential failed with the error: {e.message}"
                )
                raise DialException(
                    "Authentication failed", 401, "Unauthorized"
                )

        return self._access_token.token


async def _close_token_provider(provider: _AzureTokenProvider):
    await provider.aclose()


@cache(_close_token_provider)
def get_azure_token_provider() -> _AzureTokenProvider:
    return _AzureTokenProvider()


class OpenAICreds(TypedDict, total=False):
    api_key: str
    azure_ad_token: str


async def get_credentials(
    request_headers: Mapping[str, str],
    *,
    vendor: Vendor,
) -> OpenAICreds:
    api_key = request_headers.get("X-UPSTREAM-KEY")
    if api_key is not None:
        return {"api_key": api_key}

    match vendor:
        case Vendor.AZURE:
            return {
                "azure_ad_token": await get_azure_token_provider().get_token()
            }
        case Vendor.AWS:
            return {}
        case Vendor.VLLM | Vendor.OPENAI_PLATFORM:
            raise DialException(
                "X-UPSTREAM-KEY header is missing", 401, "Unauthorized"
            )
        case _:
            assert_never(vendor)
