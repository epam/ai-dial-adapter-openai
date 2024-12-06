from aidial_sdk.exceptions import InvalidRequestError
from fastapi import FastAPI, Request

from aidial_adapter_openai.app_config import ApplicationConfig


def set_app_config(app: FastAPI, app_config: ApplicationConfig):
    app.state.app_config = app_config


def get_app_config(app: FastAPI) -> ApplicationConfig:
    return app.state.app_config


def get_request_app_config(request: Request) -> ApplicationConfig:
    return get_app_config(request.app)


def get_api_version(request: Request) -> str:
    api_version = request.query_params.get("api-version", "")
    app_config = get_request_app_config(request)
    api_version = app_config.API_VERSIONS_MAPPING.get(api_version, api_version)

    if api_version == "":
        raise InvalidRequestError("api-version is a required query parameter")

    return api_version
