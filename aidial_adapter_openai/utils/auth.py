import os
import time
from typing import Mapping, Optional

from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity import DefaultAzureCredential
from fastapi import Request
from pydantic import BaseModel

from aidial_adapter_openai.utils.exceptions import HTTPException
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
    now = int(time.time()) - EXPIRATION_WINDOW_IN_SEC
    global access_token

    if access_token is None or now > access_token.expires_on:
        try:
            access_token = default_credential.get_token(AZURE_OPEN_AI_SCOPE)
        except ClientAuthenticationError as e:
            logger.error(
                f"Default Azure credential failed with the error: {e.message}"
            )
            raise HTTPException("Authentication failed", 401, "Unauthorized")

    return access_token.token


async def get_credentials(request: Request) -> tuple[str, str]:
    api_key = request.headers.get("X-UPSTREAM-KEY")
    if api_key is None:
        return "azure_ad", await get_api_key()
    else:
        return "azure", api_key


def get_auth_header(api_type: str, api_key: str) -> dict[str, str]:
    if api_type == "azure":
        return {"Api-Key": api_key}
    else:
        return {"Authorization": f"Bearer {api_key}"}


class Auth(BaseModel):
    name: str
    value: str

    @property
    def headers(self) -> dict[str, str]:
        return {self.name: self.value}

    @classmethod
    def from_headers(
        cls, name: str, headers: Mapping[str, str]
    ) -> Optional["Auth"]:
        value = headers.get(name)
        if value is None:
            return None
        return cls(name=name, value=value)
