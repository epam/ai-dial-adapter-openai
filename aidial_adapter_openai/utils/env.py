import os
from typing import Optional


def get_env(name: str, err_msg: Optional[str] = None) -> str:
    if name in os.environ:
        val = os.environ.get(name)
        if val is not None:
            return val

    raise Exception(err_msg or f"{name} env variable is not set")


def get_env_bool(name: str, default: bool = False) -> bool:
    return os.getenv(name, str(default)).lower() == "true"
