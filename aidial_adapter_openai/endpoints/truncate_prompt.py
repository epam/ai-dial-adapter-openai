from dataclasses import dataclass
from typing import Protocol

from aidial_adapter_anthropic.adapter import ChatCompletionAdapter
from aidial_adapter_anthropic.dial.request import ModelParameters
from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptResult,
    TruncatePromptSuccess,
)
from aidial_sdk.exceptions import RequestValidationError, ResourceNotFoundError
from anthropic import AsyncAnthropicFoundry
from fastapi import Request
from pydantic import ValidationError

from aidial_adapter_openai.chat_completions.anthropic import create_adapter
from aidial_adapter_openai.chat_completions.gpt import truncate_gpt_prompt
from aidial_adapter_openai.chat_completions.tokenizer_factory import (
    RequestTokenizer,
    create_request_tokenizer,
)
from aidial_adapter_openai.chat_completions.vllm import VllmTokenizer
from aidial_adapter_openai.chat_completions.vllm.chat_completion import (
    truncate_vllm_prompt,
)
from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    DeploymentAPIType,
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.request import (
    get_upstream_endpoint,
    get_upstream_model_name,
)
from aidial_adapter_openai.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)
from aidial_adapter_openai.utils.tokenizer import (
    Tokenizer,
    create_tiktoken_tokenizer,
)
from aidial_adapter_openai.utils.truncate_prompt import (
    truncate_prompt as truncate_prompt_with_tokenizer,
)
from aidial_adapter_openai.utils.truncation_types import DiscardedMessages
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)

# Deployment types that reuse the per-message tiktoken truncation used by
# the inline GPT chat_completion path.
_GPT_TYPES = {D.GPT4O, D.GPT4O_MINI, D.GPT_GENERIC}

# Deployment types backed by a vLLM upstream tokenize endpoint.
_VLLM_TYPES = {
    D.VLLM_CHAT_COMPLETIONS_API,
    D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API,
}

# Deployment types that reuse the Anthropic adapter's discard computation.
_ANTHROPIC_TYPES = {D.ANTHROPIC_MESSAGES_API}

# Remaining chat-like deployment types that don't have a dedicated inline
# truncation path but can be truncated by re-tokenizing the whole request
# via their request tokenizer.
_REQUEST_TOKENIZER_TYPES = {
    D.RESPONSES_API,
    D.MISTRAL,
    D.DATABRICKS,
    D.COMPLETIONS_API,
}

_SUPPORTED_TYPES = (
    _GPT_TYPES | _VLLM_TYPES | _ANTHROPIC_TYPES | _REQUEST_TOKENIZER_TYPES
)


class Truncator(Protocol):
    """Truncates a single prepared chat request using pre-built dependencies."""

    async def truncate(
        self,
        *,
        request_dict: dict,
        max_prompt_tokens: int,
        input_request: ChatCompletionRequest,
    ) -> DiscardedMessages: ...


@dataclass
class _RequestTokenizerAdapter:
    """Adapts a ``RequestTokenizer`` to the ``truncate_prompt`` interface."""

    tokenizer: RequestTokenizer

    async def tokenize(self, request: dict) -> int:
        return await self.tokenizer.tokenize_request(request)


@dataclass
class _GptTruncator:
    tokenizer: Tokenizer
    file_storage: FileStorage | None

    async def truncate(
        self,
        *,
        request_dict: dict,
        max_prompt_tokens: int,
        input_request: ChatCompletionRequest,
    ) -> DiscardedMessages:
        _, discarded, _ = await truncate_gpt_prompt(
            request=request_dict,
            file_storage=self.file_storage,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=self.tokenizer,
        )
        return discarded


@dataclass
class _VllmTruncator:
    tokenizer: VllmTokenizer
    file_storage: FileStorage | None

    async def truncate(
        self,
        *,
        request_dict: dict,
        max_prompt_tokens: int,
        input_request: ChatCompletionRequest,
    ) -> DiscardedMessages:
        _, discarded, _ = await truncate_vllm_prompt(
            request=request_dict,
            file_storage=self.file_storage,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=self.tokenizer,
        )
        return discarded


@dataclass
class _AnthropicTruncator:
    adapter: ChatCompletionAdapter

    async def truncate(
        self,
        *,
        request_dict: dict,
        max_prompt_tokens: int,
        input_request: ChatCompletionRequest,
    ) -> DiscardedMessages:
        params = ModelParameters.create(input_request)
        discarded = await self.adapter.compute_discarded_messages(
            params, input_request.messages
        )
        return discarded or []


@dataclass
class _RequestTokenizerTruncator:
    tokenizer: _RequestTokenizerAdapter

    async def truncate(
        self,
        *,
        request_dict: dict,
        max_prompt_tokens: int,
        input_request: ChatCompletionRequest,
    ) -> DiscardedMessages:
        _, discarded, _ = await truncate_prompt_with_tokenizer(
            tokenizer=self.tokenizer,
            original_request=request_dict,
            messages=request_dict["messages"],
            get_raw_message=lambda m: m,
            max_prompt_tokens=max_prompt_tokens,
        )
        return discarded


async def _load_truncate_prompt_request(
    request: Request, deployment_id: str
) -> TruncatePromptRequest:
    try:
        return await TruncatePromptRequest.from_request(
            request, deployment_id, base_url=None
        )
    except ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        msg = f"Invalid request. Path: '{path}', error: {error['msg']}"
        raise RequestValidationError(msg) from e


async def _create_truncator(
    *,
    request: Request,
    deployment_id: str,
    deployment: DeploymentAPIType,
    deployment_type: D,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: dict[str, str],
    file_storage: FileStorage | None,
    api_key: str,
) -> Truncator:
    if deployment_type in _GPT_TYPES:
        return _GptTruncator(
            tokenizer=create_tiktoken_tokenizer(
                app_config, deployment_id, deployment_type
            ),
            file_storage=file_storage,
        )

    if deployment_type in _VLLM_TYPES:
        return _VllmTruncator(
            tokenizer=VllmTokenizer(
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
            ),
            file_storage=file_storage,
        )

    if deployment_type in _ANTHROPIC_TYPES:
        vendor = app_config.get_vendor(deployment_id, deployment.endpoint)
        creds = await get_credentials(request.headers, vendor=vendor)
        api_version = get_api_version(request)
        client = deployment.endpoint.get_client(
            {**creds, "api_version": api_version, "headers": extra_headers}
        )
        if not isinstance(client, AsyncAnthropicFoundry):
            raise ValueError(
                f"Unexpected client for Anthropic deployment - {type(client)}"
            )
        model_name = get_upstream_model_name(
            request_headers=request.headers,
            deployment_id=deployment_id,
            model=None,
        )
        adapter = await create_adapter(model_name, api_key, client)
        return _AnthropicTruncator(adapter=adapter)

    tokenizer_wrapper = await create_request_tokenizer(
        request=request,
        deployment_id=deployment_id,
        deployment=deployment,
        app_config=app_config,
        upstream_endpoint=upstream_endpoint,
        extra_headers=extra_headers,
        file_storage=file_storage,
    )
    return _RequestTokenizerTruncator(
        tokenizer=_RequestTokenizerAdapter(tokenizer_wrapper)
    )


async def _truncate_prompt_input(
    *,
    request: Request,
    deployment_id: str,
    truncator: Truncator,
    input_request: ChatCompletionRequest,
) -> DiscardedMessages:
    if input_request.max_prompt_tokens is None:
        raise RequestValidationError(
            "max_prompt_tokens is required for the truncate_prompt endpoint"
        )
    max_prompt_tokens = input_request.max_prompt_tokens

    request_dict = input_request.model_dump(exclude_none=True)
    request_dict["model"] = get_upstream_model_name(
        request_headers=request.headers,
        deployment_id=deployment_id,
        model=request_dict.get("model"),
    )
    # max_prompt_tokens is passed explicitly to the truncation algorithm and
    # must not leak into the request sent to upstream tokenizers.
    request_dict.pop("max_prompt_tokens", None)

    return await truncator.truncate(
        request_dict=request_dict,
        max_prompt_tokens=max_prompt_tokens,
        input_request=input_request,
    )


async def truncate_prompt(
    deployment_id: str, request: Request
) -> TruncatePromptResponse:
    truncate_prompt_request = await _load_truncate_prompt_request(
        request, deployment_id
    )

    app_config = get_request_app_config(request)
    upstream_endpoint = get_upstream_endpoint(request.headers)
    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )
    deployment_type = deployment.deployment_type

    if deployment_type not in _SUPPORTED_TYPES:
        raise ResourceNotFoundError(
            "The truncate_prompt endpoint is not implemented for this deployment"
        )

    extra_headers = get_upstream_extra_headers(request.headers)
    file_storage = create_file_storage(request.headers)
    truncator = await _create_truncator(
        request=request,
        deployment_id=deployment_id,
        deployment=deployment,
        deployment_type=deployment_type,
        app_config=app_config,
        upstream_endpoint=upstream_endpoint,
        extra_headers=extra_headers,
        file_storage=file_storage,
        api_key=truncate_prompt_request.api_key,
    )

    outputs: list[TruncatePromptResult] = []
    for inp in truncate_prompt_request.inputs:
        try:
            discarded = await _truncate_prompt_input(
                request=request,
                deployment_id=deployment_id,
                truncator=truncator,
                input_request=inp,
            )
            outputs.append(TruncatePromptSuccess(discarded_messages=discarded))
        except Exception as e:
            outputs.append(TruncatePromptError(error=str(e)))

    return TruncatePromptResponse(outputs=outputs)
