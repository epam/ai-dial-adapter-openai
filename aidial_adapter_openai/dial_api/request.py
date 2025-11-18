from typing import Any, Type, TypeVar

from aidial_sdk.exceptions import RequestValidationError
from pydantic import BaseModel, ValidationError

_T = TypeVar("_T", bound=BaseModel)


def parse_configuration(cls: Type[_T], data: Any) -> _T | None:
    if (cf := data.get("custom_fields")) is None:
        return None

    if (conf := cf.get("configuration")) is None:
        return None

    try:
        return cls.parse_obj(conf)
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
