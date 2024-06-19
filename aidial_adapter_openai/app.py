import json
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Dict, TypeVar

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from aidial_sdk.utils.errors import json_error
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from openai import APIConnectionError, APIStatusError, APITimeoutError

from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.gpt import gpt_chat_completion
from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    gpt4_vision_chat_completion,
    gpt4o_chat_completion,
)
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.globals import http_client
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.parsers import (
    embeddings_parser,
    parse_body,
    parse_deployment_list,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.tokens import Tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Application shutdown")
    await http_client.aclose()


app = FastAPI(lifespan=lifespan)


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
gpt4o_deployments = parse_deployment_list(os.getenv("GPT4O_DEPLOYMENTS") or "")
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
        raise DialException(
            "Request timed out",
            504,
            "timeout",
            display_message="Request timed out. Please try again later.",
        )
    except APIConnectionError:
        raise DialException(
            "Error communicating with OpenAI",
            502,
            "connection",
            display_message="OpenAI server is not responsive. Please try again later.",
        )


def get_api_version(request: Request):
    api_version = request.query_params.get("api-version", "")
    api_version = api_versions_mapping.get(api_version, api_version)

    if api_version == "":
        raise DialException(
            "api-version is a required query parameter",
            400,
            "invalid_request_error",
        )

    return api_version


@app.post("/openai/deployments/{deployment_id}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):
    data = await parse_body(request)

    # Azure OpenAI deployments ignore "model" request field,
    # since the deployment id is already encoded in the endpoint path.
    # This is not the case for non-Azure OpenAI deployments, so
    # they require the "model" field to be set.
    # However, openai==1.33.0 requires the "model" field for **both**
    # Azure and non-Azure deployments.
    # Therefore, we provide the "model" field for all deployments here.
    # The same goes for /embeddings endpoint.
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

    openai_model_name = model_aliases.get(deployment_id, deployment_id)
    tokenizer = Tokenizer(model=openai_model_name)

    if deployment_id in gpt4o_deployments:
        storage = create_file_storage("images", request.headers)
        return await handle_exceptions(
            gpt4o_chat_completion(
                data,
                deployment_id,
                upstream_endpoint,
                creds,
                is_stream,
                storage,
                api_version,
                tokenizer,
            )
        )

    return await handle_exceptions(
        gpt_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            creds,
            api_version,
            tokenizer,
        )
    )


@app.post("/openai/deployments/{deployment_id}/embeddings")
async def embedding(deployment_id: str, request: Request):
    data = await parse_body(request)

    # See note for /chat/completions endpoint
    data["model"] = deployment_id

    creds = await get_credentials(request)
    api_version = get_api_version(request)
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    client = embeddings_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version}
    )

    return await handle_exceptions(
        call_with_extra_body(client.embeddings.create, data)
    )


@app.exception_handler(DialException)
def exception_handler(request: Request, exc: DialException):
    return JSONResponse(
        status_code=exc.status_code,
        content=json_error(
            message=exc.message,
            type=exc.type,
            param=exc.param,
            code=exc.code,
            display_message=exc.display_message,
        ),
    )


@app.get("/health")
def health():
    return {"status": "ok"}
