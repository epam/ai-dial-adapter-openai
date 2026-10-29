from collections.abc import Mapping
from typing import Annotated, TypeVar

from aidial_sdk.exceptions import (
    InternalServerError,
    InvalidRequestError,
)
from fastapi import Depends, Header, Request
from pydantic import BaseModel

from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.validation import RequestValue

_T = TypeVar("_T", bound=BaseModel)


def parse_configuration(cls: type[_T], request: RequestValue) -> _T | None:
    if (cf := request.get("custom_fields")) is None:
        return None

    if (conf := cf.get("configuration")) is None:
        return None

    return conf.to_model(cls)


def collect_message_text_content(message: RequestValue) -> str:
    if (content := message.get("content")) is None:
        return ""
    match content.to_str_or_list():
        case str() as text:
            return text
        case parts:
            return "".join(
                part["text"].to_str()
                for part in parts
                if part.get("type") == "text"
            )


def extract_max_prompt_tokens(request: RequestValue) -> int | None:
    # FIX_NOW: add .pop method to request to make the code as closely looking as
    # the original version
    raw_request = request.unwrap_dict()
    if (
        max_prompt_tokens := raw_request.pop("max_prompt_tokens", None)
    ) is None:
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
