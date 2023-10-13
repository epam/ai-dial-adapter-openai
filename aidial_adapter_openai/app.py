import json
import logging.config
import os

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from openai import ChatCompletion, Embedding, error

from aidial_adapter_openai.openai_override import OpenAIException
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.log_config import LogConfig
from aidial_adapter_openai.utils.parsers import (
    ApiType,
    parse_body,
    parse_upstream,
)
from aidial_adapter_openai.utils.streaming import generate_stream

logging.config.dictConfig(LogConfig().dict())
app = FastAPI()
model_aliases = json.loads(os.getenv("MODEL_ALIASES", "{}"))
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
    dial_api_key = request.headers["X-UPSTREAM-KEY"]

    api_base, upstream_deployment = parse_upstream(
        request.headers["X-UPSTREAM-ENDPOINT"], ApiType.CHAT_COMPLETION
    )

    response = await handle_exceptions(
        ChatCompletion().acreate(
            engine=upstream_deployment,
            api_key=dial_api_key,
            api_base=api_base,
            api_type="azure",
            api_version=azure_api_version,
            request_timeout=(10, 600),  # connect timeout and total timeout
            **data
        )
    )

    if is_stream:
        if isinstance(response, Response):
            return response

        return StreamingResponse(
            generate_stream(
                data["messages"],
                response,
                model_aliases.get(deployment_id, deployment_id),
                deployment_id,
            ),
            media_type="text/event-stream",
        )
    else:
        return response


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)

    dial_api_key = request.headers["X-UPSTREAM-KEY"]
    api_base, upstream_deployment = parse_upstream(
        request.headers["X-UPSTREAM-ENDPOINT"], ApiType.EMBEDDING
    )

    return await handle_exceptions(
        Embedding().acreate(
            deployment_id=upstream_deployment,
            api_key=dial_api_key,
            api_base=api_base,
            api_type="azure",
            api_version=azure_api_version,
            request_timeout=(10, 600),  # connect timeout and total timeout
            **data
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


if __name__ == "__main__":
    uvicorn.run(app, port=5000)
