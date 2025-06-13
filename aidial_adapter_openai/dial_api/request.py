from typing import Any, Type, TypeVar

from aidial_sdk.exceptions import RequestValidationError
from aidial_sdk.pydantic_v1 import BaseModel, ValidationError

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
