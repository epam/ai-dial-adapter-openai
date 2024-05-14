import json
import os
from typing import Awaitable, Dict, TypeVar

from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncStream,
)
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.gpt4_vision.chat_completion import (
    chat_completion as gpt4_vision_chat_completion,
)
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.utils.auth import OpenAICreds, get_credentials
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import configure_loggers
from aidial_adapter_openai.utils.parsers import (
    chat_completions_parser,
    embeddings_parser,
    parse_body,
    parse_deployment_list,
)
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.streaming import generate_stream, map_stream
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages

app = FastAPI()

init_telemetry(app, TelemetryConfig())
configure_loggers()

model_aliases: Dict[str, str] = json.loads(os.getenv("MODEL_ALIASES", "{}"))
dalle3_deployments = parse_deployment_list(
    os.getenv("DALLE3_DEPLOYMENTS") or ""
)
gpt4_vision_deployments = parse_deployment_list(
    os.getenv("GPT4_VISION_DEPLOYMENTS") or ""
)
mistral_deployments = parse_deployment_list(
    os.getenv("MISTRAL_DEPLOYMENTS") or ""
)
databricks_deployments = parse_deployment_list(
    os.getenv("DATABRICKS_DEPLOYMENTS") or ""
)
api_versions_mapping: Dict[str, str] = json.loads(
    os.getenv("API_VERSIONS_MAPPING", "{}")
)
dalle3_azure_api_version = os.getenv("DALLE3_AZURE_API_VERSION", "2024-02-01")

T = TypeVar("T")


async def handle_exceptions(call: Awaitable[T]) -> T | Response:
    try:
        return await call
    except APIStatusError as e:
        r = e.response
        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=r.headers,
        )
    except APITimeoutError:
        raise HTTPException("Request timed out", 504, "timeout")
    except APIConnectionError:
        raise HTTPException(
            "Error communicating with OpenAI", 502, "connection"
        )


def get_api_version(request: Request):
    api_version = request.query_params.get("api-version", "")
    api_version = api_versions_mapping.get(api_version, api_version)

    if api_version == "":
        raise HTTPException(
            "Api version is a required query parameter",
            400,
            "invalid_request_error",
        )

    return api_version


@app.post("/openai/deployments/{deployment_id}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):
    data = await parse_body(request)
    data["model"] = deployment_id

    is_stream = data.get("stream", False)

    creds = await get_credentials(request)

    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    if deployment_id in dalle3_deployments:
        storage = create_file_storage("images", request.headers)
        return await dalle3_chat_completion(
            data,
            upstream_endpoint,
            creds,
            is_stream,
            storage,
            dalle3_azure_api_version,
        )

    if deployment_id in mistral_deployments:
        return await handle_exceptions(
            mistral_chat_completion(data, upstream_endpoint, creds)
        )

    if deployment_id in databricks_deployments:
        return await handle_exceptions(
            databricks_chat_completion(data, upstream_endpoint, creds)
        )

    api_version = get_api_version(request)

    if deployment_id in gpt4_vision_deployments:
        storage = create_file_storage("images", request.headers)
        return await gpt4_vision_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            creds,
            is_stream,
            storage,
            api_version,
        )

    return await gpt_chat_completion(
        data, deployment_id, upstream_endpoint, creds, api_version
    )


async def gpt_chat_completion(
    data: dict,
    deployment_id: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    api_version: str,
):
    openai_model_name = model_aliases.get(deployment_id, deployment_id)
    tokenizer = Tokenizer(model=openai_model_name)

    discarded_messages = None
    if "max_prompt_tokens" in data:
        max_prompt_tokens = data["max_prompt_tokens"]
        if not isinstance(max_prompt_tokens, int):
            raise HTTPException(
                f"'{max_prompt_tokens}' is not of type 'integer' - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        if max_prompt_tokens < 1:
            raise HTTPException(
                f"'{max_prompt_tokens}' is less than the minimum of 1 - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        del data["max_prompt_tokens"]

        data["messages"], discarded_messages = discard_messages(
            tokenizer, data["messages"], max_prompt_tokens
        )

    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion | Response = (
        await handle_exceptions(client.chat.completions.create(**data))
    )

    if isinstance(response, Response):
        return response

    elif isinstance(response, AsyncStream):
        prompt_tokens = tokenizer.calculate_prompt_tokens(data["messages"])
        chunk_stream = map_stream(lambda obj: obj.to_dict(), response)
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
            return response.to_dict() | {
                "statistics": {"discarded_messages": discarded_messages}
            }
        else:
            return response


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)
    data["model"] = deployment_id

    creds = await get_credentials(request)
    api_version = get_api_version(request)
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    client = embeddings_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    return await handle_exceptions(client.embeddings.create(**data))


@app.exception_handler(HTTPException)
def exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": exc.message,
                "type": exc.type,
                "param": exc.param,
                "code": exc.code,
            }
        },
    )


@app.get("/health")
def health():
    return {"status": "ok"}
