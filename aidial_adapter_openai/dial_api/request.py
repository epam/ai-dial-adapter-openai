from typing import Any, Type, TypeVar

from aidial_sdk.exceptions import InvalidRequestError
from aidial_sdk.pydantic_v1 import BaseModel, ValidationError

_T = TypeVar("_T", bound=BaseModel)


def get_configuration(cls: Type[_T], data: Any) -> _T | None:
    if (cf := data.get("custom_fields")) is None:
        return None

    if (conf := cf.get("configuration")) is None:
        return None

    try:
        return cls.parse_obj(conf)
    except ValidationError as e:
        raise InvalidRequestError(f"Invalid configuration: {e!r}")
