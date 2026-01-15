import re
from json import JSONDecodeError
from typing import Any, Dict, TypedDict

from aidial_sdk.exceptions import InvalidRequestError
from anthropic import AsyncAnthropicFoundry
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncOpenAI, Timeout

from aidial_adapter_openai.utils.http_client import (
    get_anthropic_httpx_client,
    get_http_client,
)
from aidial_adapter_openai.utils.pydantic import ExtraForbidModel


class OpenAIParams(TypedDict, total=False):
    base_url: str
    api_key: str
    azure_ad_token: str
    api_version: str
    timeout: Timeout


# Retries are handled on the DIAL Core side
_MAX_RETRIES = 0


class AzureOpenAIEndpoint(ExtraForbidModel):
    azure_base_url: str | None = None
    azure_endpoint: str | None = None
    azure_deployment: str | None = None

    def get_client(self, params: OpenAIParams) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            base_url=self.azure_base_url,  # type: ignore
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment,
            api_key=params.get("api_key"),
            azure_ad_token=params.get("azure_ad_token"),
            api_version=params.get("api_version"),
            timeout=params.get("timeout"),
            max_retries=_MAX_RETRIES,
            http_client=get_http_client(),
        )


class OpenAIEndpoint(ExtraForbidModel):
    base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        api_key = params.get("api_key") or params.get("azure_ad_token")

        return AsyncOpenAI(
            base_url=self.base_url,
            api_key=api_key,
            timeout=params.get("timeout"),
            max_retries=_MAX_RETRIES,
            http_client=get_http_client(),
        )


class AnthropicEndpoint(ExtraForbidModel):
    base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncAnthropicFoundry:
        if (token := params.get("azure_ad_token")) is not None:

            def _provider():
                return token

            token_provider = _provider
        else:
            token_provider = None

        return AsyncAnthropicFoundry(
            base_url=self.base_url,
            api_key=params.get("api_key"),
            azure_ad_token_provider=token_provider,
            max_retries=_MAX_RETRIES,
            http_client=get_anthropic_httpx_client(),
        )


def _parse_endpoint(
    name: str | None, endpoint: str
) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
    if name is not None:
        suffix = f"/{name}"
        if not endpoint.endswith(suffix):
            return None
        endpoint = endpoint.removesuffix(suffix)

    # Last generation API
    if match := re.fullmatch("(.+?)/openai/deployments/(.+)", endpoint):
        return AzureOpenAIEndpoint(
            azure_endpoint=match[1], azure_deployment=match[2]
        )

    # Solely for Response API (last generation)
    # https://github.com/MicrosoftDocs/azure-ai-docs/commit/a8f37469a3ce23fcee32512d37068a9627b470d3#diff-a379a6b7b341ba9e352ab413b170b27e33790719a8631fff5d3eeac4d0a33b26R147-R153
    if match := re.fullmatch("(.+?)/openai", endpoint):
        return AzureOpenAIEndpoint(azure_endpoint=match[1])

    # Falling back to the Next generation API (aka v1 API):
    # https://learn.microsoft.com/en-us/azure/ai-foundry/openai/api-version-lifecycle?tabs=key#v1-api-support
    return OpenAIEndpoint(base_url=endpoint)


class EndpointParser(ExtraForbidModel):
    name: str | None

    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
        return _parse_endpoint(self.name, endpoint)

    def parse(self, endpoint: str) -> AzureOpenAIEndpoint | OpenAIEndpoint:
        if result := self.try_parse(endpoint):
            return result
        raise InvalidRequestError("Invalid upstream endpoint format")


class CompletionsParser(ExtraForbidModel):
    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | None:
        if "/chat/completions" in endpoint:
            return None

        return _parse_endpoint("completions", endpoint)


class AnthropicMessagesParser:
    def try_parse(self, endpoint: str) -> AnthropicEndpoint | None:
        if match := re.match(r"(.*/anthropic)/v1/messages", endpoint):
            base_url = match.group(1)
            return AnthropicEndpoint(base_url=base_url)
        return None


chat_completions_parser = EndpointParser(name="chat/completions")
image_gen_parser = EndpointParser(name="images/generations")
speech_parser = EndpointParser(name="audio/speech")
transcriptions_parser = EndpointParser(name="audio/transcriptions")
embeddings_parser = EndpointParser(name="embeddings")
responses_parser = EndpointParser(name="responses")
no_endpoint_parser = EndpointParser(name=None)
completions_parser = CompletionsParser()
azure_video_api_parser = EndpointParser(name="video/generations")
anthropic_messages_parser = AnthropicMessagesParser()


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
