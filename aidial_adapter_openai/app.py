from contextlib import asynccontextmanager
from typing import Awaitable, TypeVar

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI, Request
from fastapi.responses import Response
from openai import APIConnectionError, APIStatusError, APITimeoutError

from aidial_adapter_openai.completions import chat_completion as completion
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
    NON_STREAMING_DEPLOYMENTS,
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
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.parsers import (
    completions_parser,
    embeddings_parser,
    parse_body,
)
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.storage import create_file_storage
from aidial_adapter_openai.utils.streaming import create_server_response
from aidial_adapter_openai.utils.tokenizer import (
    MultiModalTokenizer,
    PlainTextTokenizer,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Application shutdown")
    await get_http_client().aclose()


app = FastAPI(lifespan=lifespan)


init_telemetry(app, TelemetryConfig())
configure_loggers()

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
            status_code=504,
            type="timeout",
            message="Request timed out",
            display_message="Request timed out. Please try again later.",
        )
    except APIConnectionError:
        raise DialException(
            status_code=502,
            type="connection",
            message="Error communicating with OpenAI",
            display_message="OpenAI server is not responsive. Please try again later.",
        )


def get_api_version(request: Request):
    api_version = request.query_params.get("api-version", "")
    api_version = API_VERSIONS_MAPPING.get(api_version, api_version)

    if api_version == "":
        raise InvalidRequestError("api-version is a required query parameter")

    return api_version


@app.post("/openai/deployments/{deployment_id:path}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):

    async def func():
        data = await parse_body(request)

        is_stream = bool(data.get("stream"))

        emulate_streaming = (
            deployment_id in NON_STREAMING_DEPLOYMENTS and is_stream
        )

        if emulate_streaming:
            data["stream"] = False

        return create_server_response(
            emulate_streaming,
            await run_chat_completion(deployment_id, data, is_stream, request),
        )

    return await handle_exceptions(func())


async def run_chat_completion(
    deployment_id: str, data: dict, is_stream: bool, request: Request
):

    # Azure OpenAI deployments ignore "model" request field,
    # since the deployment id is already encoded in the endpoint path.
    # This is not the case for non-Azure OpenAI deployments, so
    # they require the "model" field to be set.
    # However, openai==1.33.0 requires the "model" field for **both**
    # Azure and non-Azure deployments.
    # Therefore, we provide the "model" field for all deployments here.
    # The same goes for /embeddings endpoint.
    data["model"] = deployment_id

    creds = await get_credentials(request)
    api_version = get_api_version(request)

    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    if completions_endpoint := completions_parser.parse(upstream_endpoint):
        return await completion(
            data,
            completions_endpoint,
            creds,
            api_version,
            deployment_id,
        )

    if deployment_id in DALLE3_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await dalle3_chat_completion(
            data,
            upstream_endpoint,
            creds,
            is_stream,
            storage,
            DALLE3_AZURE_API_VERSION,
        )

    if deployment_id in MISTRAL_DEPLOYMENTS:
        return await mistral_chat_completion(data, upstream_endpoint, creds)

    if deployment_id in DATABRICKS_DEPLOYMENTS:
        return await databricks_chat_completion(data, upstream_endpoint, creds)

    if deployment_id in GPT4_VISION_DEPLOYMENTS:
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

    openai_model_name = MODEL_ALIASES.get(deployment_id, deployment_id)
    if deployment_id in GPT4O_DEPLOYMENTS:
        tokenizer = MultiModalTokenizer(openai_model_name)
        storage = create_file_storage("images", request.headers)
        return await gpt4o_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            creds,
            is_stream,
            storage,
            api_version,
            tokenizer,
        )

    tokenizer = PlainTextTokenizer(model=openai_model_name)
    return await gpt_chat_completion(
        data,
        deployment_id,
        upstream_endpoint,
        creds,
        api_version,
        tokenizer,
    )


@app.post("/openai/deployments/{deployment_id:path}/embeddings")
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
    return exc.to_fastapi_response()


@app.get("/health")
def health():
    return {"status": "ok"}
