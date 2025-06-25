import re
from json import JSONDecodeError
from typing import Any, Dict, TypedDict

from aidial_sdk.exceptions import InvalidRequestError
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncOpenAI, Timeout
from pydantic import BaseModel

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client


class OpenAIParams(TypedDict, total=False):
    api_key: str
    azure_ad_token: str
    api_version: str
    timeout: Timeout


# Retries are handled on the DIAL Core side
_MAX_RETRIES = 0


class AzureOpenAIEndpoint(BaseModel):
    azure_endpoint: str
    azure_deployment: str | None = None
    next_gen_api: bool = False

    def get_client(self, params: OpenAIParams) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=params.get("api_key"),
            azure_ad_token=params.get("azure_ad_token"),
            api_version=(
                "preview" if self.next_gen_api else params.get("api_version")
            ),
            timeout=params.get("timeout"),
            max_retries=_MAX_RETRIES,
            http_client=get_http_client(),
        )

    def get_auth_headers(self, creds: OpenAICreds) -> dict[str, str]:
        if key := creds.get("api_key"):
            return {"api-key": key}

        if token := creds.get("azure_ad_token"):
            return {"Authorization": f"Bearer {token}"}

        raise ValueError("Invalid credentials")


class OpenAIEndpoint(BaseModel):
    base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=params.get("api_key"),
            timeout=params.get("timeout"),
            max_retries=_MAX_RETRIES,
            http_client=get_http_client(),
        )

    def get_auth_headers(self, creds: OpenAICreds) -> dict[str, str]:
        if key := (creds.get("api_key") or creds.get("azure_ad_token")):
            return {"Authorization": f"Bearer {key}"}
        raise ValueError("Invalid credentials")


def _parse_endpoint(
    name, endpoint
) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
    if match := re.search(f"(.+?)/openai/deployments/(.+?)/{name}", endpoint):
        return AzureOpenAIEndpoint(
            azure_endpoint=match[1],
            azure_deployment=match[2],
        )
    if match := re.search(f"(.+?)/openai/{name}", endpoint):
        return AzureOpenAIEndpoint(
            azure_endpoint=match[1],
        )
    if match := re.search(f"(.+?)/openai/v1/{name}", endpoint):
        return AzureOpenAIEndpoint(
            azure_endpoint=match[1],
            next_gen_api=True,
        )
    if match := re.search(f"(.+?)/{name}", endpoint):
        return OpenAIEndpoint(base_url=match[1])
    return None


class EndpointParser(BaseModel):
    name: str

    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
        return _parse_endpoint(self.name, endpoint)

    def parse(self, endpoint: str) -> AzureOpenAIEndpoint | OpenAIEndpoint:
        if result := self.try_parse(endpoint):
            return result
        raise InvalidRequestError("Invalid upstream endpoint format")


class CompletionsParser(BaseModel):
    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
        if "/chat/completions" in endpoint:
            return None

        return _parse_endpoint("completions", endpoint)


chat_completions_parser = EndpointParser(name="chat/completions")
image_gen_parser = EndpointParser(name="images/generations")
embeddings_parser = EndpointParser(name="embeddings")
responses_parser = EndpointParser(name="responses")
completions_parser = CompletionsParser()


async def parse_body(request: Request) -> Dict[str, Any]:
    try:
        data = await request.json()
    except JSONDecodeError as e:
        raise InvalidRequestError(
            "Your request contained invalid JSON: " + str(e)
        )

    if not isinstance(data, dict):
        raise InvalidRequestError(str(data) + " is not of type 'object'")

    return data
