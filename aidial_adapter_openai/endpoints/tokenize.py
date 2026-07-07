from dataclasses import dataclass
from typing import Protocol, assert_never

from aidial_sdk.chat_completion.request import ChatCompletionRequest
from aidial_sdk.deployment.tokenize import (
    TokenizeError,
    TokenizeInput,
    TokenizeInputRequest,
    TokenizeOutput,
    TokenizeRequest,
    TokenizeResponse,
    TokenizeSuccess,
)
from aidial_sdk.exceptions import RequestValidationError
from fastapi import Request
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from pydantic import ValidationError

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
from aidial_adapter_openai.dial_api.request import (
    get_upstream_endpoint,
    get_upstream_model_name,
)
from aidial_adapter_openai.dial_api.storage import (
    FileStorage,
    create_file_storage,
)
from aidial_adapter_openai.responses.tokenizer import ResponsesTokenizer
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


class _Tokenizer(Protocol):
    async def tokenize_text(self, model_name: str, text: str) -> int: ...

    async def tokenize_request(self, request: dict) -> int: ...


@dataclass
class _VllmTokenizer:
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
class _TiktokenTokenizer:
    file_storage: FileStorage | None
    tokenizer: Tokenizer

    async def tokenize_text(self, model_name: str, text: str) -> int:
        return await self.tokenizer.tokenize_text(text)

    async def tokenize_request(self, request: dict) -> int:
        messages = await ResourceProcessor(
            file_storage=self.file_storage
        ).transform_messages(request["messages"])
        return await self.tokenizer.tokenize_request(request, messages)


@dataclass
class _ResponsesTokenizer:
    tokenizer: ResponsesTokenizer

    async def tokenize_text(self, model_name: str, text: str) -> int:
        return await self.tokenizer.tokenize_text(model_name, text)

    async def tokenize_request(self, request: dict) -> int:
        return await self.tokenizer.tokenize_request(request)


def _prepare_chat_request(
    value: ChatCompletionRequest, model_name: str
) -> dict:
    request = value.model_dump(exclude_none=True)
    request["model"] = model_name
    return request


async def _get_tokenizer(
    *,
    request: Request,
    deployment_id: str,
    deployment: DeploymentAPIType,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: dict[str, str],
    file_storage: FileStorage | None,
) -> _Tokenizer:
    deployment_type = deployment.deployment_type
    match deployment_type:
        case (
            D.VLLM_CHAT_COMPLETIONS_API | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            vllm_tokenizer = VllmTokenizer(
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
            )
            return _VllmTokenizer(file_storage, vllm_tokenizer)

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

            responses_tokenizer = ResponsesTokenizer(
                client=client, file_storage=file_storage
            )
            return _ResponsesTokenizer(responses_tokenizer)

        case _:
            tiktoken_model = app_config.TIKTOKEN_MODEL_MAPPING.get(
                deployment_id, deployment_id
            )
            tiktoken_tokenizer = Tokenizer(
                model=tiktoken_model,
                image_tokenizer=get_image_tokenizer(deployment_type),
            )
            return _TiktokenTokenizer(file_storage, tiktoken_tokenizer)


async def _tokenize_input(
    *,
    tokenize_input: TokenizeInput,
    tokenizer: _Tokenizer,
    model_name: str,
) -> int:
    match tokenize_input.type:
        case "string":
            return await tokenizer.tokenize_text(
                model_name, tokenize_input.value
            )
        case "request":
            request = _prepare_chat_request(tokenize_input.value, model_name)
            return await tokenizer.tokenize_request(request)
        case unreachable:
            assert_never(unreachable)


async def _load_tokenize_request(
    request: Request, deployment_id: str
) -> TokenizeRequest:
    try:
        return await TokenizeRequest.from_request(
            request, deployment_id, base_url=None
        )
    except ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        msg = f"Invalid request. Path: '{path}', error: {error['msg']}"
        raise RequestValidationError(msg) from e


async def tokenize(deployment_id: str, request: Request) -> TokenizeResponse:
    tokenize_request = await _load_tokenize_request(request, deployment_id)

    app_config = get_request_app_config(request)
    upstream_endpoint = get_upstream_endpoint(request.headers)
    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )

    extra_headers = get_upstream_extra_headers(request.headers)
    file_storage = create_file_storage(request.headers)

    tokenizer = await _get_tokenizer(
        request=request,
        deployment_id=deployment_id,
        deployment=deployment,
        app_config=app_config,
        upstream_endpoint=upstream_endpoint,
        extra_headers=extra_headers,
        file_storage=file_storage,
    )

    outputs: list[TokenizeOutput] = []
    for tokenize_input in tokenize_request.inputs:
        try:
            request_model = (
                tokenize_input.value.model
                if isinstance(tokenize_input, TokenizeInputRequest)
                else None
            )
            model_name = get_upstream_model_name(
                request_headers=request.headers,
                deployment_id=deployment_id,
                model=request_model,
            )
            token_count = await _tokenize_input(
                tokenize_input=tokenize_input,
                tokenizer=tokenizer,
                model_name=model_name,
            )
            outputs.append(TokenizeSuccess(token_count=token_count))
        except Exception as exc:
            outputs.append(TokenizeError(error=str(exc)))

    return TokenizeResponse(outputs=outputs)
