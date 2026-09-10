from collections.abc import Mapping
from typing import Annotated, Any, TypeVar

from aidial_sdk.exceptions import (
    InternalServerError,
    InvalidRequestError,
    RequestValidationError,
)
from fastapi import Depends, Header, Request
from pydantic import BaseModel, ValidationError

from aidial_adapter_openai.utils.log_config import logger

_T = TypeVar("_T", bound=BaseModel)


def parse_configuration(cls: type[_T], data: Any) -> _T | None:
    if (cf := data.get("custom_fields")) is None:
        return None

    if (conf := cf.get("configuration")) is None:
        return None

    try:
        return cls.model_validate(conf)
    except ValidationError as e:
        error = e.errors()[0]
        path = ".".join(map(str, error["loc"]))
        msg = f"Invalid request. Path: 'custom_field.configuration.{path}', error: {error['msg']}"

        raise RequestValidationError(msg)


def collect_message_text_content(message: dict) -> str:
    text = ""
    if content := message.get("content"):
        if isinstance(content, str):
            text += content
        elif isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text += item["text"]
    return text


def extract_max_prompt_tokens(request: dict) -> int | None:
    if (max_prompt_tokens := request.pop("max_prompt_tokens", None)) is None:
        return None

    if not isinstance(max_prompt_tokens, int):
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is not of type 'integer'",
            param="max_prompt_tokens",
        )

    if max_prompt_tokens < 1:
        raise InvalidRequestError(
            f"'{max_prompt_tokens}' is less than the minimum of 1",
            param="max_prompt_tokens",
        )

    return max_prompt_tokens


def _resolve_deployment_id(
    request: Request,
    override_name: Annotated[
        str | None, Header(alias="X-DIAL-OVERRIDE-NAME")
    ] = None,
) -> str:
    # Resolving upstream deployment ID via a series of fallbacks.
    # The key principle - is not to trust whatever came in the request model field,
    # since it isn't something that could be controlled in the DIAL Core config.
    upstream_deployment_id = (
        override_name
        or request.path_params.get("deployment_id")
        or _get_dial_deployment_id(request.headers)
    )
    logger.debug(f"Upstream deployment ID: {upstream_deployment_id}")
    return upstream_deployment_id


# The deployment id path parameter of an Azure-style endpoint,
# with the override name applied.
DeploymentId = Annotated[str, Depends(_resolve_deployment_id)]


def _get_dial_deployment_id(request_headers: Mapping[str, str]) -> str:
    name = "X-DIAL-DEPLOYMENT-ID"
    if (deployment_id := request_headers.get(name)) is None:
        raise InternalServerError(f"{name} header is missing in the request.")

    logger.debug(f"DIAL deployment ID: {deployment_id}")
    return deployment_id


def get_upstream_endpoint(request_headers: Mapping[str, str]) -> str:
    name = "X-UPSTREAM-ENDPOINT"
    if (endpoint := request_headers.get(name)) is None:
        raise InternalServerError(f"{name} header is missing in the request.")

    logger.debug(f"upstream endpoint: {endpoint}")
    return endpoint
