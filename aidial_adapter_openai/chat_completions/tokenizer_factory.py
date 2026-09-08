"""Factory for request/text tokenizers keyed by deployment type.

Shared by the ``tokenize`` and ``truncate_prompt`` endpoints so both derive
the tokenizer for a deployment through a single implementation.
"""

from dataclasses import dataclass
from typing import Protocol, assert_never

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.exceptions import ResourceNotFoundError
from anthropic import AsyncAnthropic
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

from aidial_adapter_openai.chat_completions.anthropic import create_adapter
from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm.chat_completion import (
    transform_vllm_messages,
)
from aidial_adapter_openai.chat_completions.vllm.tokenizer import (
    VllmTokenizer,
)
from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    DeploymentAPIType,
    Vendor,
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.tokenizer import ResponsesRequestTokenizer
from aidial_adapter_openai.utils.client import get_client
from aidial_adapter_openai.utils.request import get_api_version
from aidial_adapter_openai.utils.tokenizer import (
    Tokenizer,
    create_tiktoken_tokenizer,
)


class RequestTokenizer(Protocol):
    """Tokenizes either a piece of text or a full chat completion request."""

    async def tokenize_text(self, model_name: str, text: str) -> int: ...

    async def tokenize_request(self, request: ChatCompletionRequest) -> int: ...

    async def tokenize_raw_request(self, request: dict) -> int: ...


@dataclass
class _VllmRequestTokenizer:
    file_storage: FileStorage | None
    tokenizer: VllmTokenizer

    async def tokenize_text(self, model_name: str, text: str) -> int:
        return await self.tokenizer.tokenize(
            {
                "model": model_name,
                "prompt": text,
                "add_special_tokens": False,
            }
        )

    async def tokenize_request(self, request: ChatCompletionRequest) -> int:
        return await self.tokenize_raw_request(
            request.model_dump(exclude_none=True)
        )

    async def tokenize_raw_request(self, request: dict) -> int:
        request["messages"] = await transform_vllm_messages(
            request["messages"], self.file_storage
        )
        return await self.tokenizer.tokenize(request)


@dataclass
class _TiktokenRequestTokenizer:
    file_storage: FileStorage | None
    tokenizer: Tokenizer

    async def tokenize_text(self, model_name: str, text: str) -> int:
        return await self.tokenizer.tokenize_text(text)

    async def tokenize_request(self, request: ChatCompletionRequest) -> int:
        return await self.tokenize_raw_request(
            request.model_dump(exclude_none=True)
        )

    async def tokenize_raw_request(self, request: dict) -> int:
        messages = await ResourceProcessor(
            file_storage=self.file_storage
        ).transform_messages(request["messages"])
        return await self.tokenizer.tokenize_request(request, messages)

    async def tokenize(self, request: dict) -> int:
        return await self.tokenize_raw_request(request)


@dataclass
class _AnthropicRequestTokenizer:
    adapter: ChatCompletionAdapter

    async def tokenize_text(self, model_name: str, text: str) -> int:
        return await self.adapter.count_completion_tokens(text)

    async def tokenize_request(self, request: ChatCompletionRequest) -> int:
        params = ModelParameters.create(request)
        return await self.adapter.count_prompt_tokens(params, request.messages)

    async def tokenize_raw_request(self, request: dict) -> int:
        raise ValueError("Raw request tokenization isn't supported")


async def create_request_tokenizer(
    *,
    request: Request,
    deployment_id: str,
    deployment: DeploymentAPIType,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: dict[str, str],
    file_storage: FileStorage | None,
    api_key: str,
    vendor: Vendor,
) -> RequestTokenizer:
    deployment_type = deployment.deployment_type

    def _tiktoken_tokenizer() -> _TiktokenRequestTokenizer:
        tokenizer = create_tiktoken_tokenizer(
            app_config, deployment_id, deployment_type
        )
        return _TiktokenRequestTokenizer(file_storage, tokenizer)

    match deployment_type:
        case (
            D.VLLM_CHAT_COMPLETIONS_API | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            vllm_tokenizer = VllmTokenizer(
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
            )
            return _VllmRequestTokenizer(file_storage, vllm_tokenizer)

        case D.RESPONSES_API:
            client = await get_client(
                request=request,
                deployment_id=deployment_id,
                deployment=deployment,
                app_config=app_config,
                extra_headers=extra_headers,
                api_version=get_api_version(request),
            )
            if not isinstance(
                client, AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI
            ):
                raise ValueError(
                    f"Unexpected client for the deployment backed by Responses API - {type(client)}"
                )

            match vendor:
                case Vendor.OPENAI_PLATFORM:
                    return ResponsesRequestTokenizer(client, file_storage)
                case Vendor.AWS | Vendor.AZURE | Vendor.ALIBABA:
                    raise ResourceNotFoundError(
                        "The tokenize and truncate_prompt endpoints are not "
                        "implemented for Responses API deployments backed by "
                        "Azure OpenAI, Amazon Bedrock or Alibaba Cloud "
                        "Model Studio."
                    )
                case Vendor.VLLM:
                    raise ValueError(
                        "Unexpected vendor backed by Responses API - VLLM."
                    )
                case _:
                    assert_never(client)

        case D.ANTHROPIC_MESSAGES_API:
            client = await get_client(
                request=request,
                deployment_id=deployment_id,
                deployment=deployment,
                app_config=app_config,
                extra_headers=extra_headers,
                api_version=get_api_version(request),
            )
            if not isinstance(client, AsyncAnthropic):
                raise ValueError(
                    f"Unexpected client for Anthropic deployment - {type(client)}"
                )
            adapter = await create_adapter(deployment_id, api_key, client)
            return _AnthropicRequestTokenizer(adapter=adapter)

        case _:
            return _tiktoken_tokenizer()
