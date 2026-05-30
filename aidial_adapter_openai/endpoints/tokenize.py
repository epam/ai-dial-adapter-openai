from collections.abc import Mapping
from typing import Any, Literal

from aidial_sdk.exceptions import InvalidRequestError
from fastapi import Request
from typing_extensions import TypedDict

from aidial_adapter_openai.chat_completions.transformation import (
    ResourceProcessor,
)
from aidial_adapter_openai.chat_completions.vllm.chat_completion import (
    transform_vllm_messages,
)
from aidial_adapter_openai.chat_completions.vllm.tokenizer import (
    VllmTokenizer,
)
from aidial_adapter_openai.configuration.app_config import ApplicationConfig
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
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.parsers import parse_body
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.tokenizer import Tokenizer
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


class TokenizeSuccessOutput(TypedDict):
    status: Literal["success"]
    token_count: int


class TokenizeErrorOutput(TypedDict):
    status: Literal["error"]
    error: str


TokenizeOutput = TokenizeSuccessOutput | TokenizeErrorOutput


class TokenizeResponse(TypedDict):
    outputs: list[TokenizeOutput]


def _parse_inputs(body: dict) -> list[dict]:
    inputs = body.get("inputs")
    if not isinstance(inputs, list):
        raise InvalidRequestError(
            "'inputs' must be a list",
            param="inputs",
        )
    return inputs


def _validate_input_item(item: Any, index: int) -> tuple[str, Any]:
    if not isinstance(item, dict):
        raise ValueError(f"inputs[{index}] must be an object")

    input_type = item.get("type")
    if input_type not in ("request", "string"):
        raise ValueError(
            f"inputs[{index}].type must be 'request' or 'string', got {input_type!r}"
        )

    if "value" not in item:
        raise ValueError(f"inputs[{index}].value is required")

    value = item["value"]
    if input_type == "string":
        if not isinstance(value, str):
            raise ValueError(f"inputs[{index}].value must be a string")
    elif not isinstance(value, dict):
        raise ValueError(f"inputs[{index}].value must be an object")

    return input_type, value


def _prepare_chat_request(value: dict, model_name: str) -> dict:
    request = {**value, "model": value.get("model") or model_name}
    if "messages" not in request:
        raise ValueError("'messages' field is required for request input")
    return request


async def _tokenize_input(
    *,
    input_type: str,
    value: Any,
    deployment_id: str,
    model_name: str,
    deployment_type: D,
    app_config: ApplicationConfig,
    upstream_endpoint: str,
    extra_headers: Mapping[str, str],
    file_storage: FileStorage | None,
) -> int:
    match deployment_type:
        case (
            D.VLLM_CHAT_COMPLETIONS_API | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            tokenizer = VllmTokenizer(
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
            )
            if input_type == "string":
                return await tokenizer.tokenize(
                    {
                        "model": model_name,
                        "prompt": value,
                        "add_special_tokens": False,
                    }
                )

            request = _prepare_chat_request(value, model_name)
            request["messages"] = await transform_vllm_messages(
                request["messages"], file_storage
            )
            return await tokenizer.tokenize(request)

        case _:
            tiktoken_model = app_config.TIKTOKEN_MODEL_MAPPING.get(
                deployment_id, deployment_id
            )
            tokenizer = Tokenizer(
                model=tiktoken_model,
                image_tokenizer=get_image_tokenizer(deployment_type),
            )
            if input_type == "string":
                return await tokenizer.tokenize_text(value)

            request = _prepare_chat_request(value, model_name)
            messages = await ResourceProcessor(
                file_storage=file_storage
            ).transform_messages(request["messages"])
            return await tokenizer.tokenize_request(request, messages)


async def tokenize(deployment_id: str, request: Request) -> TokenizeResponse:
    app_config = get_request_app_config(request)
    body = await parse_body(request)
    inputs = _parse_inputs(body)

    upstream_endpoint = get_upstream_endpoint(request.headers)
    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )
    deployment_type = deployment.deployment_type

    extra_headers = get_upstream_extra_headers(request.headers)
    file_storage = create_file_storage(request.headers)

    outputs: list[TokenizeOutput] = []
    for index, item in enumerate(inputs):
        try:
            input_type, value = _validate_input_item(item, index)
            request_model = (
                value.get("model") if input_type == "request" else None
            )
            model_name = get_upstream_model_name(
                request.headers,
                deployment_id,
                model=request_model,
            )
            token_count = await _tokenize_input(
                input_type=input_type,
                value=value,
                deployment_id=deployment_id,
                model_name=model_name,
                deployment_type=deployment_type,
                app_config=app_config,
                upstream_endpoint=upstream_endpoint,
                extra_headers=extra_headers,
                file_storage=file_storage,
            )
            outputs.append({"status": "success", "token_count": token_count})
        except Exception as exc:
            outputs.append({"status": "error", "error": str(exc)})

    return {"outputs": outputs}
