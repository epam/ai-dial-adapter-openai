import dataclasses
import json
from typing import Any


def _get_dict(data: Any) -> dict:
    try:
        return data.__dict__
    except AttributeError:
        return {}


def to_json(data: Any, drill_down: bool = False) -> Any:
    if isinstance(data, dict):
        return {k: to_json(v, drill_down) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [to_json(i, drill_down) for i in data]
    else:
        try:
            json.dumps(data)
            return data
        except (TypeError, OverflowError):
            try:
                return dataclasses.asdict(data)
            except TypeError:
                _str = str(data)
                d = _get_dict(data)
                if d and drill_down:
                    return {"@@this": _str, **to_json(d, drill_down)}
                else:
                    return _str
