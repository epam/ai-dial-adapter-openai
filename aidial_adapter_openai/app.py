from contextlib import asynccontextmanager

import pydantic
from aidial_sdk._errors import pydantic_validation_exception_handler
from aidial_sdk.exceptions import HTTPException as DialException
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

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.routers.chat_completion import chat_completion
from aidial_adapter_openai.routers.embeddings import embedding
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.request import set_app_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Application shutdown")
    await get_http_client().aclose()


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


def pydantic_exception_handler(request: Request, exc: pydantic.ValidationError):
    return pydantic_validation_exception_handler(request, exc)


def dial_exception_handler(request: Request, exc: DialException):
    return exc.to_fastapi_response()


def create_app(
    app_config: ApplicationConfig | None = None,
    to_init_telemetry: bool = True,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    set_app_config(app, app_config or ApplicationConfig.from_env())

    if to_init_telemetry:
        init_telemetry(app, TelemetryConfig())

    configure_loggers()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    app.post("/openai/deployments/{deployment_id:path}/embeddings")(embedding)
    app.post("/openai/deployments/{deployment_id:path}/chat/completions")(
        chat_completion
    )
    app.exception_handler(OpenAIError)(openai_exception_handler)
    app.exception_handler(pydantic.ValidationError)(pydantic_exception_handler)
    app.exception_handler(DialException)(dial_exception_handler)

    return app


app = create_app()
