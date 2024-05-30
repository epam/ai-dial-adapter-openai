import json
import os
from typing import Dict

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import Request
from fastapi.responses import JSONResponse

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.gpt import gpt_chat_completion
from aidial_adapter_openai.gpt4_code_interpreter.application import (
    CodeInterpreterApplication,
)
from aidial_adapter_openai.gpt4_vision.chat_completion import (
    chat_completion as gpt4_vision_chat_completion,
)
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.utils.auth import get_credentials_from_request
from aidial_adapter_openai.utils.exceptions import (
    HTTPException,
    handle_exceptions,
)
from aidial_adapter_openai.utils.log_config import configure_loggers
from aidial_adapter_openai.utils.parsers import (
    embeddings_parser,
    parse_body,
    parse_deployment_list,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.tokens import Tokenizer
from aidial_adapter_openai.utils.version import get_api_version_from_request

app = DIALApp(add_healthcheck=True, telemetry_config=TelemetryConfig())

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
dalle3_azure_api_version = os.getenv("DALLE3_AZURE_API_VERSION", "2024-02-01")

# FIXME: remove the non-trivial default, it only for the review env experimentation
gpt4_code_interpreter_deployments = parse_deployment_list(
    os.getenv("GPT_CODE_INTERPRETER_DEPLOYMENTS") or "gpt-4-0613"
)

for deployment_id in gpt4_code_interpreter_deployments:
    app.add_chat_completion(
        deployment_id, CodeInterpreterApplication(deployment_id)
    )


@app.post("/openai/deployments/{deployment_id}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):

    data = await parse_body(request)
    data["model"] = deployment_id

    is_stream = data.get("stream", False)

    creds = await get_credentials_from_request(request)

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

    api_version = get_api_version_from_request(request)

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

    openai_model_name = model_aliases.get(deployment_id, deployment_id)
    tokenizer = Tokenizer(model=openai_model_name)

    return await handle_exceptions(
        gpt_chat_completion(
            data,
            deployment_id,
            tokenizer,
            upstream_endpoint,
            creds,
            api_version,
        )
    )


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)
    data["model"] = deployment_id

    creds = await get_credentials_from_request(request)
    api_version = get_api_version_from_request(request)
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    client = embeddings_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    return await handle_exceptions(
        call_with_extra_body(client.embeddings.create, data)
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
