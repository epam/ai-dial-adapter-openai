import json
import os
from typing import Callable, Dict, List, Optional, TypeVar

from aidial_adapter_openai.utils.log_config import logger


def get_env(name: str, err_msg: Optional[str] = None) -> str:
    if (val := os.getenv(name)) is not None:
        return val
    raise Exception(err_msg or f"{name} env variable is not set")


def get_env_bool(name: str, default: bool = False) -> bool:
    if (value := os.getenv(name)) is not None:
        return value.lower() == "true"
    return default


def get_env_list(name: str) -> List[str] | None:
    if (value := os.getenv(name)) is not None:
        return list(map(str.strip, (value).split(",")))
    return None


def get_env_dict(key: str) -> Dict[str, str] | None:
    if (value := os.getenv(key)) is not None:
        try:
            return json.loads(value)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Environment variable {key!r} is not a valid JSON: {value!r}"
            ) from e
    return None


_T = TypeVar("_T")


def get_env_var(
    parser: Callable[[str], _T],
    name: str,
    *,
    deprecated_names: List[str] | None = None,
) -> _T:
    for alt in deprecated_names or []:
        if os.getenv(alt) is not None:
            logger.warning(
                f"{alt} environment variable is deprecated. Use {name} instead."
            )
            return parser(alt)
    return parser(name)
