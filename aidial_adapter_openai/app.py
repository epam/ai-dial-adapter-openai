from contextlib import asynccontextmanager

import pydantic
from aidial_sdk._errors import pydantic_validation_exception_handler
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InvalidRequestError
from aidial_sdk.telemetry.init import init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI, Request
from fastapi.responses import Response
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAIError,
)

from aidial_adapter_openai.completions import chat_completion as completion
from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.embeddings.azure_ai_vision import (
    embeddings as azure_ai_vision_embeddings,
)
from aidial_adapter_openai.embeddings.openai import (
    embeddings as openai_embeddings,
)
from aidial_adapter_openai.env import (
    API_VERSIONS_MAPPING,
    AZURE_AI_VISION_DEPLOYMENTS,
    DALLE3_AZURE_API_VERSION,
    DALLE3_DEPLOYMENTS,
    DATABRICKS_DEPLOYMENTS,
    GPT4_VISION_DEPLOYMENTS,
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
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.parsers import completions_parser, parse_body
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


def get_api_version(request: Request):
    api_version = request.query_params.get("api-version", "")
    api_version = API_VERSIONS_MAPPING.get(api_version, api_version)

    if api_version == "":
        raise InvalidRequestError("api-version is a required query parameter")

    return api_version


@app.post("/openai/deployments/{deployment_id:path}/chat/completions")
async def chat_completion(deployment_id: str, request: Request):

    data = await parse_body(request)

    is_stream = bool(data.get("stream"))

    emulate_streaming = deployment_id in NON_STREAMING_DEPLOYMENTS and is_stream

    if emulate_streaming:
        data["stream"] = False

    return create_server_response(
        emulate_streaming,
        await call_chat_completion(deployment_id, data, is_stream, request),
    )


async def call_chat_completion(
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

    text_tokenizer_model = MODEL_ALIASES.get(deployment_id, deployment_id)

    if image_tokenizer := get_image_tokenizer(deployment_id):
        storage = create_file_storage("images", request.headers)

        if deployment_id in GPT4_VISION_DEPLOYMENTS:
            tokenizer = MultiModalTokenizer("gpt-4", image_tokenizer)
            return await gpt4_vision_chat_completion(
                data,
                deployment_id,
                upstream_endpoint,
                creds,
                is_stream,
                storage,
                api_version,
                tokenizer,
            )
        else:
            tokenizer = MultiModalTokenizer(
                text_tokenizer_model, image_tokenizer
            )
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

    tokenizer = PlainTextTokenizer(model=text_tokenizer_model)
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

    if deployment_id in AZURE_AI_VISION_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await azure_ai_vision_embeddings(
            creds, deployment_id, upstream_endpoint, storage, data
        )

    return await openai_embeddings(creds, upstream_endpoint, api_version, data)


@app.exception_handler(OpenAIError)
def openai_exception_handler(request: Request, e: DialException):
    if isinstance(e, APIStatusError):
        r = e.response
        headers = r.headers

        # Avoid encoding the error message when the original response was encoded.
        if "Content-Encoding" in headers:
            del headers["Content-Encoding"]

        return Response(
            content=r.content,
            status_code=r.status_code,
            headers=headers,
        )

    if isinstance(e, APITimeoutError):
        raise DialException(
            status_code=504,
            type="timeout",
            message="Request timed out",
            display_message="Request timed out. Please try again later.",
        )

    if isinstance(e, APIConnectionError):
        raise DialException(
            status_code=502,
            type="connection",
            message="Error communicating with OpenAI",
            display_message="OpenAI server is not responsive. Please try again later.",
        )

    if isinstance(e, APIError):
        raise DialException(
            status_code=getattr(e, "status_code", None) or 500,
            message=e.message,
            type=e.type,
            code=e.code,
            param=e.param,
            display_message=None,
        )


@app.exception_handler(pydantic.ValidationError)
def pydantic_exception_handler(request: Request, exc: pydantic.ValidationError):
    return pydantic_validation_exception_handler(request, exc)


@app.exception_handler(DialException)
def dial_exception_handler(request: Request, exc: DialException):
    return exc.to_fastapi_response()


@app.get("/health")
def health():
    return {"status": "ok"}
