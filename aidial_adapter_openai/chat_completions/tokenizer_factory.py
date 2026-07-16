"""Factory for request/text tokenizers keyed by deployment type.

Shared by the ``tokenize`` and ``truncate_prompt`` endpoints so both derive
the tokenizer for a deployment through a single implementation.
"""

from dataclasses import dataclass
from typing import Protocol

from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

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
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.tokenizer import ResponsesTokenizer
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.request import get_api_version
from aidial_adapter_openai.utils.tokenizer import (
    Tokenizer,
    create_tiktoken_tokenizer,
)


class RequestTokenizer(Protocol):
    """Tokenizes either a piece of text or a full chat completion request."""

    async def tokenize_text(self, model_name: str, text: str) -> int: ...

    async def tokenize_request(self, request: dict) -> int: ...


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

    async def tokenize_request(self, request: dict) -> int:
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

    async def tokenize_request(self, request: dict) -> int:
        messages = await ResourceProcessor(
            file_storage=self.file_storage
        ).transform_messages(request["messages"])
        return await self.tokenizer.tokenize_request(request, messages)


async def create_request_tokenizer(
    *,
    request: Request,
    deployment_id: str,
    deployment: DeploymentAPIType,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: dict[str, str],
    file_storage: FileStorage | None,
) -> RequestTokenizer:
    deployment_type = deployment.deployment_type
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
            deployment_endpoint = deployment.endpoint
            vendor = app_config.get_vendor(deployment_id, deployment_endpoint)
            creds = await get_credentials(request.headers, vendor=vendor)
            api_version = get_api_version(request)
            client = deployment_endpoint.get_client(
                {**creds, "api_version": api_version, "headers": extra_headers}
            )

            if not isinstance(
                client, AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI
            ):
                raise ValueError(
                    f"Unexpected client for the deployment backed by Responses API - {type(client)}"
                )

            return ResponsesTokenizer(client=client, file_storage=file_storage)

        case _:
            tiktoken_tokenizer = create_tiktoken_tokenizer(
                app_config, deployment_id, deployment_type
            )
            return _TiktokenRequestTokenizer(file_storage, tiktoken_tokenizer)
