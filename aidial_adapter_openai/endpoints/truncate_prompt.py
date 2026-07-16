from dataclasses import dataclass

from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.truncate_prompt import (
    TruncatePromptError,
    TruncatePromptRequest,
    TruncatePromptResponse,
    TruncatePromptResult,
    TruncatePromptSuccess,
)
from aidial_sdk.exceptions import RequestValidationError, ResourceNotFoundError
from fastapi import Request
from pydantic import ValidationError

from aidial_adapter_openai.chat_completions.gpt import (
    multi_modal_truncate_prompt,
)
from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm import VllmTokenizer
from aidial_adapter_openai.chat_completions.vllm.chat_completion import (
    transform_vllm_messages,
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
from aidial_adapter_openai.endpoints.tokenize import _get_tokenizer, _Tokenizer
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.truncate_prompt import (
    truncate_prompt as coarse_truncate_prompt,
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

# Remaining chat-like deployment types that don't have a dedicated inline
# truncation path but can be truncated coarsely via their tokenizer.
_COARSE_TYPES = {
    D.RESPONSES_API,
    D.MISTRAL,
    D.DATABRICKS,
    D.COMPLETIONS_API,
}

_SUPPORTED_TYPES = _GPT_TYPES | _VLLM_TYPES | _COARSE_TYPES


@dataclass
class _CoarseTokenizerAdapter:
    """Bridges the ``_Tokenizer`` wrapper to the coarse truncation interface."""

    tokenizer: _Tokenizer

    async def tokenize(self, request: dict) -> int:
        return await self.tokenizer.tokenize_request(request)


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


async def _truncate_prompt_input(
    *,
    request: Request,
    deployment_id: str,
    deployment: DeploymentAPIType,
    deployment_type: D,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: dict[str, str],
    file_storage: FileStorage | None,
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

    messages: list[dict] = request_dict["messages"]

    if deployment_type in _GPT_TYPES:
        tiktoken_model = app_config.TIKTOKEN_MODEL_MAPPING.get(
            deployment_id, deployment_id
        )
        tokenizer = Tokenizer(
            model=tiktoken_model,
            image_tokenizer=get_image_tokenizer(deployment_type),
        )
        multi_modal_messages = await ResourceProcessor(
            file_storage=file_storage
        ).transform_messages(messages)
        _, discarded, _ = await multi_modal_truncate_prompt(
            request_dict,
            multi_modal_messages,
            max_prompt_tokens,
            tokenizer,
        )
        return discarded

    if deployment_type in _VLLM_TYPES:
        vllm_tokenizer = VllmTokenizer(
            upstream_endpoint=upstream_endpoint,
            extra_headers=extra_headers,
        )
        transformed = await transform_vllm_messages(messages, file_storage)
        _, discarded, _ = await coarse_truncate_prompt(
            tokenizer=vllm_tokenizer,
            original_request=request_dict,
            messages=transformed,
            get_raw_message=lambda m: m,
            max_prompt_tokens=max_prompt_tokens,
        )
        return discarded

    tokenizer_wrapper = await _get_tokenizer(
        request=request,
        deployment_id=deployment_id,
        deployment=deployment,
        app_config=app_config,
        upstream_endpoint=upstream_endpoint,
        extra_headers=extra_headers,
        file_storage=file_storage,
    )
    _, discarded, _ = await coarse_truncate_prompt(
        tokenizer=_CoarseTokenizerAdapter(tokenizer_wrapper),
        original_request=request_dict,
        messages=messages,
        get_raw_message=lambda m: m,
        max_prompt_tokens=max_prompt_tokens,
    )
    return discarded


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

    outputs: list[TruncatePromptResult] = []
    for inp in truncate_prompt_request.inputs:
        try:
            discarded = await _truncate_prompt_input(
                request=request,
                deployment_id=deployment_id,
                deployment=deployment,
                deployment_type=deployment_type,
                app_config=app_config,
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
                file_storage=file_storage,
                input_request=inp,
            )
            outputs.append(TruncatePromptSuccess(discarded_messages=discarded))
        except Exception as e:
            outputs.append(TruncatePromptError(error=str(e)))

    return TruncatePromptResponse(outputs=outputs)
