import json
import os
from typing import Dict, List, Optional


def get_env(name: str, err_msg: Optional[str] = None) -> str:
    if name in os.environ:
        val = os.environ.get(name)
        if val is not None:
            return val

    raise Exception(err_msg or f"{name} env variable is not set")


def get_env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"


def get_env_list(name: str) -> List[str] | None:
    value = os.getenv(name)
    if value is None:
        return None
    return list(map(str.strip, (value).split(",")))


def get_env_dict(key: str) -> Dict[str, str] | None:
    value = os.getenv(key)
    return json.loads(value) if value else None
