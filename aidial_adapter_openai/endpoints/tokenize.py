from typing import assert_never

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
from pydantic import ValidationError

from aidial_adapter_openai.dial_api.request import (
    get_upstream_endpoint,
    get_upstream_model_name,
)
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.utils.request import get_request_app_config
from aidial_adapter_openai.utils.tokenizer_factory import (
    RequestTokenizer,
    create_request_tokenizer,
)
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


async def _tokenize_input(
    *,
    tokenize_input: TokenizeInput,
    tokenizer: RequestTokenizer,
    model_name: str,
) -> int:
    match tokenize_input.type:
        case "string":
            return await tokenizer.tokenize_text(
                model_name, tokenize_input.value
            )
        case "request":
            request = tokenize_input.value
            request.model = model_name
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

    tokenizer = await create_request_tokenizer(
        request=request,
        deployment_id=deployment_id,
        deployment=deployment,
        app_config=app_config,
        upstream_endpoint=upstream_endpoint,
        extra_headers=extra_headers,
        file_storage=file_storage,
        api_key=tokenize_request.api_key,
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
