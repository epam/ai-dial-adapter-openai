import re
from collections.abc import Callable
from http import HTTPStatus
from json import JSONDecodeError
from typing import Any
from urllib.parse import urlparse

from aidial_sdk.exceptions import HTTPException, InvalidRequestError
from anthropic import AsyncAnthropic, AsyncAnthropicFoundry
from aws_bedrock_token_generator import provide_token
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from typing_extensions import TypedDict

from aidial_adapter_openai.utils.http_client import (
    get_anthropic_httpx_client,
    get_http_client,
)
from aidial_adapter_openai.utils.pydantic import ExtraForbidModel


class OpenAIParams(TypedDict, total=False):
    base_url: str
    api_key: str
    azure_ad_token: str
    api_version: str | None
    headers: dict[str, str]


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
            max_retries=_MAX_RETRIES,
            default_headers=params.get("headers"),
            http_client=get_http_client(),
        )


class OpenAIEndpoint(ExtraForbidModel):
    openai_base_url: str

    def get_client(self, params: OpenAIParams) -> AsyncOpenAI:
        api_key = params.get("api_key") or params.get("azure_ad_token")

        return AsyncOpenAI(
            base_url=self.openai_base_url,
            api_key=api_key,
            max_retries=_MAX_RETRIES,
            default_headers=params.get("headers"),
            http_client=get_http_client(),
        )


class AnthropicEndpoint(ExtraForbidModel):
    anthropic_base_url: str
    # Azure AI Foundry endpoints require Azure-specific authentication,
    # while other providers of the Messages API (Anthropic itself,
    # Fireworks, OpenRouter etc) are served by the vanilla client.
    foundry: bool

    def get_client(self, params: OpenAIParams) -> AsyncAnthropic:
        if not self.foundry:
            return AsyncAnthropic(
                base_url=self.anthropic_base_url,
                api_key=params.get("api_key"),
                max_retries=_MAX_RETRIES,
                http_client=get_anthropic_httpx_client(),
            )

        if (token := params.get("azure_ad_token")) is not None:

            def _provider():
                return token

            token_provider = _provider
        else:
            token_provider = None

        return AsyncAnthropicFoundry(
            base_url=self.anthropic_base_url,
            api_key=params.get("api_key"),
            azure_ad_token_provider=token_provider,
            max_retries=_MAX_RETRIES,
            http_client=get_anthropic_httpx_client(),
        )


class BedrockOpenAIEndpoint(ExtraForbidModel):
    bedrock_region: str

    def get_client(self, params: OpenAIParams) -> AsyncBedrockOpenAI:
        api_key = params.get("api_key")
        token_provider: Callable | None = None

        if not api_key:

            def _token_provider() -> str:
                return provide_token(self.bedrock_region)

            token_provider = _token_provider

        return AsyncBedrockOpenAI(
            bedrock_token_provider=token_provider,
            aws_region=self.bedrock_region,
            api_key=api_key,
            max_retries=_MAX_RETRIES,
            default_headers=params.get("headers"),
            http_client=get_http_client(),
        )


def _parse_endpoint(
    name: str | None, endpoint: str
) -> AzureOpenAIEndpoint | OpenAIEndpoint | BedrockOpenAIEndpoint | None:
    if name is not None:
        suffix = f"/{name}"
        if not endpoint.endswith(suffix):
            return None
        endpoint = endpoint.removesuffix(suffix)

    if match := re.fullmatch(
        r"https?://bedrock-mantle\.([a-z0-9-]+)\.api\.aws(?:/.*)?",
        endpoint,
    ):
        return BedrockOpenAIEndpoint(bedrock_region=match[1])

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
    return OpenAIEndpoint(openai_base_url=endpoint)


class EndpointParser(ExtraForbidModel):
    name: str | None

    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | BedrockOpenAIEndpoint | None:
        return _parse_endpoint(self.name, endpoint)

    def parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | BedrockOpenAIEndpoint:
        if result := self.try_parse(endpoint):
            return result
        raise bad_upstream_endpoint()


class OptionalEndpointParser(ExtraForbidModel):
    name: str | None

    def parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | BedrockOpenAIEndpoint:
        result = _parse_endpoint(self.name, endpoint)
        result = result or _parse_endpoint(None, endpoint)
        if result:
            return result
        raise bad_upstream_endpoint()


class CompletionsParser(ExtraForbidModel):
    def try_parse(
        self, endpoint: str
    ) -> AzureOpenAIEndpoint | OpenAIEndpoint | BedrockOpenAIEndpoint | None:
        if "/chat/completions" in endpoint:
            return None

        return _parse_endpoint("completions", endpoint)


class AnthropicEndpointParser:
    @staticmethod
    def _is_foundry_base_url(base_url: str) -> bool:
        hostname = urlparse(base_url).hostname or ""
        return "azure" in hostname and base_url.endswith("/anthropic")

    def try_parse(self, endpoint: str) -> AnthropicEndpoint | None:
        if match := re.match(r"(.*?)/v1/messages", endpoint):
            base_url = match.group(1)
            return AnthropicEndpoint(
                anthropic_base_url=base_url,
                foundry=self._is_foundry_base_url(base_url),
            )
        return None


chat_completions_parser = EndpointParser(name="chat/completions")
chat_completions_optional_parser = OptionalEndpointParser(
    name="chat/completions"
)
image_gen_parser = EndpointParser(name="images/generations")
speech_parser = EndpointParser(name="audio/speech")
transcriptions_parser = EndpointParser(name="audio/transcriptions")
embeddings_parser = EndpointParser(name="embeddings")
responses_parser = EndpointParser(name="responses")
completions_parser = CompletionsParser()
azure_video_api_parser = EndpointParser(name="video/generations")
openai_video_api_parser = EndpointParser(name="videos")
anthropic_messages_parser = AnthropicEndpointParser()


async def parse_body(request: Request) -> dict[str, Any]:
    try:
        data = await request.json()
    except JSONDecodeError as e:
        raise InvalidRequestError(
            "Your request contained invalid JSON: " + str(e)
        )

    if not isinstance(data, dict):
        raise InvalidRequestError(str(data) + " is not of type 'object'")

    return data


def bad_upstream_endpoint(message: str = "") -> HTTPException:
    suffix = f": {message}" if message else ""
    return HTTPException(
        status_code=HTTPStatus.BAD_GATEWAY,
        type="internal_server_error",
        message=f"Invalid upstream endpoint format{suffix}",
    )
