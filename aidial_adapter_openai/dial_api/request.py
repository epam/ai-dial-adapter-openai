from collections.abc import Mapping
from typing import Any, Type, TypeVar

from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

from aidial_adapter_openai.utils.log_config import logger

_T = TypeVar("_T", bound=BaseModel)


def parse_configuration(cls: Type[_T], data: Any) -> _T | None:
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


def get_upstream_endpoint(request_headers: Mapping[str, str]) -> str:
    name = "X-UPSTREAM-ENDPOINT"
    if (endpoint := request_headers.get(name)) is None:
        raise ValueError(f"{name} header is missing in the request.")

    logger.debug(f"upstream endpoint: {endpoint}")
    return endpoint
