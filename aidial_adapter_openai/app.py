from contextlib import asynccontextmanager

import fastapi
from aidial_client import AsyncDialClientPool
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.telemetry.init import init_telemetry as sdk_init_telemetry
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import FastAPI
from openai import OpenAIError

import aidial_adapter_openai.endpoints as endpoints
from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.exceptions.handlers import (
    adapter_exception_handler,
    fastapi_exception_handler,
)
from aidial_adapter_openai.utils.auth import get_azure_token_provider
from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.http_client import (
    get_anthropic_httpx_client,
    get_http_client,
)
from aidial_adapter_openai.utils.log_config import configure_loggers, logger
from aidial_adapter_openai.utils.request import set_app_config


async def _close_dial_client_pool(pool: AsyncDialClientPool) -> None:
    await pool.aclose()


@cache(_close_dial_client_pool)
def get_dial_client_pool() -> AsyncDialClientPool:
    return AsyncDialClientPool()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    logger.info("Application shutdown")
    await get_http_client.clear()
    await get_anthropic_httpx_client.clear()
    await get_dial_client_pool.clear()
    await get_azure_token_provider.clear()


def create_app(
    app_config: ApplicationConfig | None = None,
    init_telemetry: bool = True,
) -> FastAPI:
    app = FastAPI(lifespan=lifespan)

    if init_telemetry:
        sdk_init_telemetry(app, TelemetryConfig())

    configure_loggers()

    set_app_config(app, app_config or ApplicationConfig.from_env())

    app.get("/health")(endpoints.health)

    app.post("/openai/v1/responses")(endpoints.responses_create)
    app.get("/openai/v1/responses/{responses_id:str}")(
        endpoints.responses_retrieve
    )
    app.delete("/openai/v1/responses/{responses_id:str}")(
        endpoints.responses_delete
    )
    app.post("/openai/v1/responses/{responses_id:str}/cancel")(
        endpoints.responses_cancel
    )

    app.post("/openai/deployments/{deployment_id:path}/embeddings")(
        endpoints.embedding
    )
    app.post("/openai/deployments/{deployment_id:path}/chat/completions")(
        endpoints.chat_completion
    )
    app.post("/openai/deployments/{deployment_id:path}/tokenize")(
        endpoints.tokenize
    )
    app.get("/openai/deployments/{deployment_id:path}/configuration")(
        endpoints.configuration
    )

    app.add_exception_handler(fastapi.HTTPException, fastapi_exception_handler)

    for exc_class in [OpenAIError, DialException]:
        app.add_exception_handler(exc_class, adapter_exception_handler)

    return app


app = create_app()
