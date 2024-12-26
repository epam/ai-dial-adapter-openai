from contextlib import asynccontextmanager

from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.telemetry.init import init_telemetry as sdk_init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI
from openai import OpenAIError

import aidial_adapter_openai.endpoints as endpoints
from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.exception_handlers import adapter_exception_handler
from aidial_adapter_openai.utils.http_client import get_http_client
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.request import set_app_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Application shutdown")
    await get_http_client().aclose()


def create_app(
    app_config: ApplicationConfig | None = None,
    init_telemetry: bool = True,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    set_app_config(app, app_config or ApplicationConfig.from_env())

    if init_telemetry:
        sdk_init_telemetry(app, TelemetryConfig())

    configure_loggers()

    app.get("/health")(endpoints.health)
    app.post("/openai/deployments/{deployment_id:path}/embeddings")(
        endpoints.embedding
    )
    app.post("/openai/deployments/{deployment_id:path}/chat/completions")(
        endpoints.chat_completion
    )

    for exc_class in [OpenAIError, DialException]:
        app.add_exception_handler(exc_class, adapter_exception_handler)

    return app


app = create_app()
