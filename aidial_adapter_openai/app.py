from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import ChatCompletion, Embedding
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.env import (
    API_VERSIONS_MAPPING,
    DALLE3_AZURE_API_VERSION,
    DALLE3_DEPLOYMENTS,
    DATABRICKS_DEPLOYMENTS,
    GPT4_VISION_DEPLOYMENTS,
    GPT4O_DEPLOYMENTS,
    MISTRAL_DEPLOYMENTS,
    MODEL_ALIASES,
)
from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    gpt4_vision_chat_completion,
    gpt4o_chat_completion,
)
from aidial_adapter_openai.legacy_completions import legacy_completions
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.exceptions import handle_exceptions
from aidial_adapter_openai.utils.log_config import configure_loggers
from aidial_adapter_openai.utils.parsers import (
    chat_completions_parser,
    completions_parser,
    embeddings_parser,
    parse_body,
)
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.streaming import generate_stream, map_stream
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages

app = FastAPI()

init_telemetry(app, TelemetryConfig())
configure_loggers()


def get_api_version(request: Request):
    api_version = request.query_params.get("api-version", "")
    api_version = API_VERSIONS_MAPPING.get(api_version, api_version)

    if api_version == "":
        raise DialException(
            "Api version is a required query parameter",
            400,
            "invalid_request_error",
        )

    return api_version


@app.post("/openai/deployments/{deployment_id}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):

    data = await parse_body(request)
    is_stream = bool(data.get("stream", False))
    api_version = get_api_version(request)

    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    # If it is not chat completions, but legacy completions
    if not chat_completions_parser.is_valid(
        upstream_endpoint
    ) and completions_parser.is_valid(upstream_endpoint):
        api_type, api_key = await get_credentials(request, completions_parser)

        return await handle_exceptions(
            legacy_completions(
                data,
                deployment_id,
                is_stream,
                api_key,
                api_type,
                api_version,
                **(
                    completions_parser.parse(
                        upstream_endpoint
                    ).prepare_request_args(deployment_id)
                ),
            ),
        )

    api_type, api_key = await get_credentials(request, chat_completions_parser)
    if deployment_id in DALLE3_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await dalle3_chat_completion(
            data,
            upstream_endpoint,
            api_key,
            is_stream,
            storage,
            api_type,
            DALLE3_AZURE_API_VERSION,
        )
    elif deployment_id in MISTRAL_DEPLOYMENTS:
        return await handle_exceptions(
            mistral_chat_completion(data, upstream_endpoint, api_key)
        )
    elif deployment_id in DATABRICKS_DEPLOYMENTS:
        return await handle_exceptions(
            databricks_chat_completion(
                data,
                deployment_id,
                upstream_endpoint,
                api_key,
                api_type,
            )
        )

    if deployment_id in GPT4_VISION_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await gpt4_vision_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            api_key,
            is_stream,
            storage,
            api_type,
            api_version,
        )

    openai_model_name = MODEL_ALIASES.get(deployment_id, deployment_id)
    tokenizer = Tokenizer(model=openai_model_name)

    if deployment_id in GPT4O_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await handle_exceptions(
            gpt4o_chat_completion(
                data,
                deployment_id,
                upstream_endpoint,
                api_key,
                is_stream,
                storage,
                api_type,
                api_version,
                tokenizer,
            )
        )

    discarded_messages = None
    if "max_prompt_tokens" in data:
        max_prompt_tokens = data["max_prompt_tokens"]
        if not isinstance(max_prompt_tokens, int):
            raise DialException(
                f"'{max_prompt_tokens}' is not of type 'integer' - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        if max_prompt_tokens < 1:
            raise DialException(
                f"'{max_prompt_tokens}' is less than the minimum of 1 - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        del data["max_prompt_tokens"]

        data["messages"], discarded_messages = discard_messages(
            tokenizer, data["messages"], max_prompt_tokens
        )

    request_args = chat_completions_parser.parse(
        upstream_endpoint
    ).prepare_request_args(deployment_id)

    response = await handle_exceptions(
        ChatCompletion().acreate(
            api_key=api_key,
            api_type=api_type,
            api_version=api_version,
            request_timeout=DEFAULT_TIMEOUT,
            **(data | request_args),
        )
    )

    if isinstance(response, Response):
        return response

    if is_stream:
        prompt_tokens = tokenizer.calculate_prompt_tokens(data["messages"])
        chunk_stream = map_stream(lambda obj: obj.to_dict_recursive(), response)
        return StreamingResponse(
            to_openai_sse_stream(
                generate_stream(
                    prompt_tokens,
                    chunk_stream,
                    tokenizer,
                    deployment_id,
                    discarded_messages,
                )
            ),
            media_type="text/event-stream",
        )
    else:
        if discarded_messages is not None:
            assert isinstance(response, OpenAIObject)
            response = response.to_dict() | {
                "statistics": {"discarded_messages": discarded_messages}
            }

        return response


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)

    api_type, api_key = await get_credentials(request, embeddings_parser)
    api_version = get_api_version(request)

    request_args = embeddings_parser.parse(
        request.headers["X-UPSTREAM-ENDPOINT"]
    ).prepare_request_args(deployment_id)

    return await handle_exceptions(
        Embedding().acreate(
            api_key=api_key,
            api_type=api_type,
            api_version=api_version,
            request_timeout=DEFAULT_TIMEOUT,
            **(data | request_args),
        )
    )


@app.exception_handler(DialException)
def exception_handler(request: Request, exc: DialException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "param": exc.param,
                "code": exc.code,
                "display_message": exc.display_message,
            }
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}
