import re
from abc import ABC, abstractmethod
from json import JSONDecodeError
from typing import Dict, List, Union

from aidial_sdk.exceptions import HTTPException as DialException
from fastapi import Request
from pydantic import BaseModel

from aidial_adapter_openai.types import ChatCompletionRequestData


class Endpoint(ABC):
    @abstractmethod
    def prepare_request_args(self, deployment_id: str) -> Dict[str, str]:
        pass

    @abstractmethod
    def get_api_type(self) -> str:
        pass


class AzureOpenAIEndpoint(BaseModel):
    api_base: str
    deployment_id: str

    def prepare_request_args(self, deployment_id: str) -> Dict[str, str]:
        return {"api_base": self.api_base, "engine": self.deployment_id}

    def get_api_type(self) -> str:
        return "azure"


class OpenAIEndpoint(BaseModel):
    api_base: str

    def prepare_request_args(self, deployment_id: str) -> Dict[str, str]:
        return {"api_base": self.api_base, "model": deployment_id}

    def get_api_type(self) -> str:
        return "open_ai"


class EndpointParser(BaseModel):
    name: str

    def _find_match(
        self,
        endpoint,
    ) -> Union[AzureOpenAIEndpoint, OpenAIEndpoint, None]:
        match = re.search(
            f"(.+?)/openai/deployments/(.+?)/{self.name}", endpoint
        )

        if match:
            return AzureOpenAIEndpoint(
                api_base=match[1], deployment_id=match[2]
            )

        match = re.search(f"(.+?)/{self.name}", endpoint)
        if match:
            return OpenAIEndpoint(api_base=match[1])
        return match

    def is_valid(self, endpoint: str) -> bool:
        if self._find_match(endpoint):
            return True
        return False

    def parse(self, endpoint: str) -> AzureOpenAIEndpoint | OpenAIEndpoint:
        if match := self._find_match(endpoint):
            return match

        raise DialException(
            "Invalid upstream endpoint format", 400, "invalid_request_error"
        )


chat_completions_parser = EndpointParser(name="chat/completions")
completions_parser = EndpointParser(name="completions")
embeddings_parser = EndpointParser(name="embeddings")


async def parse_body(
    request: Request,
) -> ChatCompletionRequestData:
    try:
        data = await request.json()
    except JSONDecodeError as e:
        raise DialException(
            "Your request contained invalid JSON: " + str(e),
            400,
            "invalid_request_error",
        )

    if not isinstance(data, dict):
        raise DialException(
            str(data) + " is not of type 'object'", 400, "invalid_request_error"
        )

    return data  # type: ignore


def parse_deployment_list(deployments: str) -> List[str]:
    if deployments is None:
        return []

    return list(map(str.strip, deployments.split(",")))
