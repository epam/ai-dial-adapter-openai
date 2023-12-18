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
from aidial_adapter_openai.gpt4_vision import (
    chat_completion as gpt4_vision_chat_completion,
)
from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import LogConfig
from aidial_adapter_openai.utils.parsers import (
    ApiType,
    parse_body,
    parse_upstream,
)
from aidial_adapter_openai.utils.request_classifier import (
    does_request_use_functions_or_tools,
)
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.streaming import generate_stream
from aidial_adapter_openai.utils.tokens import discard_messages
from aidial_adapter_openai.utils.versions import compare_versions

logging.config.dictConfig(LogConfig().dict())
app = FastAPI()
model_aliases: Dict[str, str] = json.loads(os.getenv("MODEL_ALIASES", "{}"))
azure_api_version = os.getenv("AZURE_API_VERSION", "2023-03-15-preview")


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


@app.post("/openai/deployments/{deployment_id}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):
    data = await parse_body(request)

    is_stream = data.get("stream", False)
    openai_model_name = model_aliases.get(deployment_id, deployment_id)
    api_key = request.headers["X-UPSTREAM-KEY"]
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    if deployment_id.lower() == "dalle3":
        file_storage = await create_file_storage("dalle", request.headers)
        return await dalle3_chat_completion(
            data, upstream_endpoint, api_key, is_stream, file_storage
        )
    elif deployment_id.lower() == "gpt-4-vision-review":
        file_storage = await create_file_storage("gpt4-v", request.headers)
        return await gpt4_vision_chat_completion(
            data, upstream_endpoint, api_key, is_stream, file_storage
        )

    api_base, upstream_deployment = parse_upstream(
        upstream_endpoint, ApiType.CHAT_COMPLETION
    )

    api_version = azure_api_version

    if does_request_use_functions_or_tools(data):
        request_api_version = request.query_params.get("api-version")

        if request_api_version is not None:
            # 2023-07-01-preview is the first azure api version that supports functions
            compare_result = compare_versions(request_api_version, "2023-07-01")
            if compare_result == 0 or compare_result == 1:
                api_version = request_api_version

    discarded_messages = None
    if "max_prompt_tokens" in data:
        max_prompt_tokens = data["max_prompt_tokens"]
        if type(max_prompt_tokens) != int:
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
            data["messages"], openai_model_name, max_prompt_tokens
        )

    response = await handle_exceptions(
        ChatCompletion().acreate(
            engine=upstream_deployment,
            api_key=api_key,
            api_base=api_base,
            api_type="azure",
            api_version=api_version,
            request_timeout=(10, 600),  # connect timeout and total timeout
            **data,
        )
    )

    if is_stream:
        if isinstance(response, Response):
            return response

        return StreamingResponse(
            generate_stream(
                data["messages"],
                response,
                openai_model_name,
                deployment_id,
                discarded_messages,
            ),
            media_type="text/event-stream",
        )
    else:
        if discarded_messages is not None:
            assert type(response) == OpenAIObject

            response_with_statistics = response.to_dict() | {
                "statistics": {"discarded_messages": discarded_messages}
            }

            return response_with_statistics

        return response


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)

    api_key = request.headers["X-UPSTREAM-KEY"]
    api_base, upstream_deployment = parse_upstream(
        request.headers["X-UPSTREAM-ENDPOINT"], ApiType.EMBEDDING
    )

    return await handle_exceptions(
        Embedding().acreate(
            deployment_id=upstream_deployment,
            api_key=api_key,
            api_base=api_base,
            api_type="azure",
            api_version=azure_api_version,
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
