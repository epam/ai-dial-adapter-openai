import json
import os
from typing import Dict

from fastapi import Request

from aidial_adapter_openai.utils.exceptions import HTTPException

api_versions_mapping: Dict[str, str] = json.loads(
    os.getenv("API_VERSIONS_MAPPING", "{}")
)


def get_api_version_from_request(request: Request):
    return get_api_version(request.query_params.get("api-version"))


def get_api_version(api_version: str | None):
    api_version = api_version or ""
    api_version = api_versions_mapping.get(api_version, api_version)

    if api_version == "":
        raise HTTPException(
            "Api version is a required query parameter",
            400,
            "invalid_request_error",
        )

    return api_version
