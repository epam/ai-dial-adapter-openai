import os
import time
from typing import Mapping

from aidial_sdk.exceptions import HTTPException as DialException
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity.aio import DefaultAzureCredential
from typing_extensions import TypedDict

from aidial_adapter_openai.utils.log_config import logger

default_credential = DefaultAzureCredential()
access_token: AccessToken | None = None

EXPIRATION_WINDOW_IN_SEC: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRATION_WINDOW", 10)
)
AZURE_OPEN_AI_SCOPE: str = os.getenv(
    "AZURE_OPEN_AI_SCOPE", "https://cognitiveservices.azure.com/.default"
)


async def get_api_key() -> str:
    now = int(time.time())
    global access_token

    if (
        access_token is None
        or now + EXPIRATION_WINDOW_IN_SEC > access_token.expires_on
    ):
        try:
            access_token = await default_credential.get_token(
                AZURE_OPEN_AI_SCOPE
            )
        except ClientAuthenticationError as e:
            logger.error(
                f"Default Azure credential failed with the error: {e.message}"
            )
            raise DialException("Authentication failed", 401, "Unauthorized")

    return access_token.token


class OpenAICreds(TypedDict, total=False):
    api_key: str
    azure_ad_token: str


async def get_credentials(request_headers: Mapping[str, str]) -> OpenAICreds:
    api_key = request_headers.get("X-UPSTREAM-KEY")
    if api_key is None:
        return {"azure_ad_token": await get_api_key()}
    else:
        return {"api_key": api_key}
