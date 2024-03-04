import time
from typing import Mapping, Optional

from azure.core.credentials import AccessToken
from azure.identity import DefaultAzureCredential
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
