from fastapi import FastAPI, Request

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.log_config import logger as log


def set_app_config(app: FastAPI, app_config: ApplicationConfig):
    log.debug(f"Setting application config: {app_config.model_dump_json()}")
    app.state.app_config = app_config


def get_app_config(app: FastAPI) -> ApplicationConfig:
    return app.state.app_config


def get_request_app_config(request: Request) -> ApplicationConfig:
    return get_app_config(request.app)


def get_api_version(request: Request) -> str | None:
    """
    The api-version query parameter is only meaningful for the versioned
    Azure OpenAI API. The endpoints which do require it, report the missing
    parameter themselves: see `AzureOpenAIEndpoint.get_client`.
    """
    api_version = request.query_params.get("api-version", "")
    app_config = get_request_app_config(request)
    api_version = app_config.API_VERSIONS_MAPPING.get(api_version, api_version)

    return api_version or None
