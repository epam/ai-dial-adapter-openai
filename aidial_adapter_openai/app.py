import json
import logging.config
import os
from typing import Dict

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import ChatCompletion, Embedding, error
from openai.openai_object import OpenAIObject

from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.gpt4_vision.chat_completion import (
    chat_completion as gpt4_vision_chat_completion,
)
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import LogConfig
from aidial_adapter_openai.utils.parsers import (
    ApiType,
    parse_body,
    parse_deployment_list,
    parse_upstream,
)
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.streaming import generate_stream, map_stream
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages

logging.config.dictConfig(LogConfig().dict())
app = FastAPI()
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
api_versions_mapping: Dict[str, str] = json.loads(
    os.getenv("API_VERSIONS_MAPPING", "{}")
)
dalle3_azure_api_version = os.getenv("DALLE3_AZURE_API_VERSION", "2024-02-01")


async def handle_exceptions(call):
    try:
        return await call
    except OpenAIException as e:
        return Response(status_code=e.code, headers=e.headers, content=e.body)
    except error.Timeout:
        raise HTTPException("Request timed out", 504, "timeout")
    except error.APIConnectionError:
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

    is_stream = data.get("stream", False)

    api_type, api_key = await get_credentials(request)

    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    if deployment_id in dalle3_deployments:
        storage = create_file_storage("images", request.headers)
        return await dalle3_chat_completion(
            data,
            upstream_endpoint,
            api_key,
            is_stream,
            storage,
            api_type,
            dalle3_azure_api_version,
        )
    elif deployment_id in mistral_deployments:
        return await handle_exceptions(
            mistral_chat_completion(data, upstream_endpoint, api_key, is_stream)
        )

    api_version = get_api_version(request)

    if deployment_id in gpt4_vision_deployments:
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

    openai_model_name = model_aliases.get(deployment_id, deployment_id)
    tokenizer = Tokenizer(model=openai_model_name)

    api_base, upstream_deployment = parse_upstream(
        upstream_endpoint, ApiType.CHAT_COMPLETION
    )

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

    response = await handle_exceptions(
        ChatCompletion().acreate(
            engine=upstream_deployment,
            api_key=api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            request_timeout=(10, 600),  # connect timeout and total timeout
            **data,
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
            assert type(response) == OpenAIObject
            response = response.to_dict() | {
                "statistics": {"discarded_messages": discarded_messages}
            }

        return response


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)

    api_type, api_key = await get_credentials(request)
    api_base, upstream_deployment = parse_upstream(
        request.headers["X-UPSTREAM-ENDPOINT"], ApiType.EMBEDDING
    )
    api_version = get_api_version(request)

    return await handle_exceptions(
        Embedding().acreate(
            deployment_id=upstream_deployment,
            api_key=api_key,
            api_base=api_base,
            api_type=api_type,
            api_version=api_version,
            request_timeout=(10, 600),  # connect timeout and total timeout
            **data,
        )
    )


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
