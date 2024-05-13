import re
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Any, Dict, List, TypedDict

from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncOpenAI, Timeout
from pydantic import BaseModel

from aidial_adapter_openai.utils.exceptions import HTTPException


class OpenAIParams(TypedDict, total=False):
    api_key: str
    azure_ad_token: str
    api_version: str
    timeout: Timeout


class Endpoint(ABC):
    @abstractmethod
    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        pass


class AzureOpenAIEndpoint(BaseModel):
    base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            base_url=self.base_url,
            api_key=params.get("api_key"),
            azure_ad_token=params.get("azure_ad_token"),
            api_version=params.get("api_version"),
            timeout=params.get("timeout"),
        )


class OpenAIEndpoint(BaseModel):
    base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=params.get("api_key"),
            timeout=params.get("timeout"),
        )


class EndpointParser(BaseModel):
    name: str

    def parse(self, endpoint: str) -> AzureOpenAIEndpoint | OpenAIEndpoint:
        match = re.search(
            f"(.+?)/openai/deployments/(.+?)/{self.name}", endpoint
        )

        if match:
            return AzureOpenAIEndpoint(
                base_url=f"{match[1]}/openai/deployments/{match[2]}"
            )

        match = re.search(f"(.+?)/{self.name}", endpoint)

        if match:
            return OpenAIEndpoint(base_url=match[1])

        raise HTTPException(
            "Invalid upstream endpoint format", 400, "invalid_request_error"
        )


chat_completions_parser = EndpointParser(name="chat/completions")
embeddings_parser = EndpointParser(name="embeddings")


async def parse_body(
    request: Request,
) -> Dict[str, Any]:
    try:
        data = await request.json()
    except JSONDecodeError as e:
        raise HTTPException(
            "Your request contained invalid JSON: " + str(e),
            400,
            "invalid_request_error",
        )

    if not isinstance(data, dict):
        raise HTTPException(
            str(data) + " is not of type 'object'", 400, "invalid_request_error"
        )

    return data


def parse_deployment_list(deployments: str) -> List[str]:
    if deployments is None:
        return []

    return list(map(str.strip, deployments.split(",")))
