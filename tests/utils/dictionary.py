from typing import Iterable


def exclude_keys(d: dict, keys: Iterable[str]) -> dict:
    return {k: v for k, v in d.items() if k not in keys}
