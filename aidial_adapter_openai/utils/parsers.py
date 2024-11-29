import re
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Dict, TypedDict

from aidial_sdk.exceptions import InvalidRequestError
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncOpenAI, Timeout
from pydantic import BaseModel

from aidial_adapter_openai.utils.http_client import get_http_client


class OpenAIParams(TypedDict, total=False):
    api_key: str
    azure_ad_token: str
    api_version: str
    timeout: Timeout


class Endpoint(ABC):
    @abstractmethod
    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        pass


# Retries are handled on the DIAL Core side
_MAX_RETRIES = 0


class AzureOpenAIEndpoint(BaseModel):
    azure_endpoint: str
    azure_deployment: str

    def get_client(self, params: OpenAIParams) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=params.get("api_key"),
            azure_ad_token=params.get("azure_ad_token"),
            api_version=params.get("api_version"),
            timeout=params.get("timeout"),
            max_retries=_MAX_RETRIES,
            http_client=get_http_client(),
        )


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


def _parse_endpoint(
    name, endpoint
) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
    if azure_match := re.search(
        f"(.+?)/openai/deployments/(.+?)/{name}", endpoint
    ):
        return AzureOpenAIEndpoint(
            azure_endpoint=azure_match[1],
            azure_deployment=azure_match[2],
        )
    elif openai_match := re.search(f"(.+?)/{name}", endpoint):
        return OpenAIEndpoint(base_url=openai_match[1])
    else:
        return None


class EndpointParser(BaseModel):
    name: str

    def parse(self, endpoint: str) -> AzureOpenAIEndpoint | OpenAIEndpoint:
        if result := _parse_endpoint(self.name, endpoint):
            return result
        raise InvalidRequestError("Invalid upstream endpoint format")


class CompletionsParser(BaseModel):
    def parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
        if "/chat/completions" in endpoint:
            return None

        return _parse_endpoint("completions", endpoint)


chat_completions_parser = EndpointParser(name="chat/completions")
embeddings_parser = EndpointParser(name="embeddings")
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
