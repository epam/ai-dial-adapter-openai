import time
from typing import Mapping, Optional

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential
from fastapi import Request
from openai import util
from openai.util import ApiType
from pydantic import BaseModel

default_credential = DefaultAzureCredential()
access_token: AccessToken | None = None

EXPIRATION_WINDOW_IN_SEC = 10


async def get_api_key():
    now = int(time.time()) - EXPIRATION_WINDOW_IN_SEC
    global access_token
    if access_token is None or now > access_token.expires_on:
        access_token = default_credential.get_token(
            "https://cognitiveservices.azure.com/.default"
        )
    return access_token.token


async def get_credentials(request: Request):
    api_key = request.headers.get("X-UPSTREAM-KEY")
    api_type = "azure"
    if api_key is None:
        api_key = await get_api_key()
        api_type = "azure_ad"
    return api_type, api_key


def get_auth_header(api_type: str, api_key: str):
    return util.api_key_to_header(ApiType.from_str(api_type), api_key)


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
