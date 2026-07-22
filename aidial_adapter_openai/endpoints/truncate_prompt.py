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


class Truncator(Protocol):
    async def truncate(
        self, max_prompt_tokens: int, request: ChatCompletionRequest
    ) -> DiscardedMessages: ...


@dataclass
class _GptTruncator:
    tokenizer: Tokenizer
    file_storage: FileStorage | None

    async def truncate(
        self, max_prompt_tokens: int, request: ChatCompletionRequest
    ) -> DiscardedMessages:
        request_dict = request.model_dump(exclude_none=True)
        _, discarded, _ = await truncate_gpt_prompt(
            request=request_dict,
            file_storage=self.file_storage,
            max_prompt_tokens=max_prompt_tokens,
            tokenizer=self.tokenizer,
        )
        return discarded


@dataclass
class _RequestTokenizerTruncator:
    tokenizer: RequestTokenizer

    async def tokenize(self, request: dict) -> int:
        return await self.tokenizer.tokenize_request(request)

    async def truncate(
        self, max_prompt_tokens: int, request: ChatCompletionRequest
    ) -> DiscardedMessages:
        request_dict = request.model_dump(exclude_none=True)
        _, discarded, _ = await truncate_prompt_with_tokenizer(
            tokenizer=self,
            original_request=request_dict,
            messages=request_dict["messages"],
            get_raw_message=lambda m: m,
            max_prompt_tokens=max_prompt_tokens,
        )
        return discarded


@dataclass
class _AnthropicTruncator:
    adapter: ChatCompletionAdapter

    async def truncate(
        self, max_prompt_tokens: int, request: ChatCompletionRequest
    ) -> DiscardedMessages:
        params = ModelParameters.create(request)
        params.max_prompt_tokens = max_prompt_tokens
        discarded = await self.adapter.compute_discarded_messages(
            params, request.messages
        )
        return discarded or []


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
    extra_headers = get_upstream_extra_headers(request.headers)
    file_storage = create_file_storage(request.headers)
    model_name = get_upstream_model_name(
        request_headers=request.headers,
        deployment_id=deployment_id,
        model=None,
    )

    truncator: Truncator
    match deployment_type:
        case D.ANTHROPIC_MESSAGES_API:
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
            adapter = await create_adapter(
                model_name, truncate_prompt_request.api_key, client
            )
            truncator = _AnthropicTruncator(adapter=adapter)
        case D.GPT4O | D.GPT4O_MINI | D.GPT_GENERIC:
            truncator = _GptTruncator(
                tokenizer=create_tiktoken_tokenizer(
                    app_config, deployment_id, deployment_type
                ),
                file_storage=file_storage,
            )
        case (
            D.RESPONSES_API
            | D.VLLM_CHAT_COMPLETIONS_API
            | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            tokenizer = await create_request_tokenizer(
                request=request,
                deployment_id=deployment_id,
                deployment=deployment,
                app_config=app_config,
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
                file_storage=file_storage,
            )
            truncator = _RequestTokenizerTruncator(tokenizer=tokenizer)
        case _ as not_implemented:
            raise ResourceNotFoundError(
                f"The truncate_prompt endpoint is not implemented for this deployment: {not_implemented}"
            )

    outputs: list[TruncatePromptResult] = []
    for inp in truncate_prompt_request.inputs:
        max_prompt_tokens = inp.max_prompt_tokens
        inp.model = model_name
        inp.max_prompt_tokens = None

        try:
            if max_prompt_tokens is None:
                raise RequestValidationError(
                    "max_prompt_tokens is required for the truncate_prompt endpoint"
                )
            discarded = await truncator.truncate(max_prompt_tokens, inp)
            outputs.append(TruncatePromptSuccess(discarded_messages=discarded))
        except Exception as e:
            outputs.append(TruncatePromptError(error=str(e)))

    return TruncatePromptResponse(outputs=outputs)
