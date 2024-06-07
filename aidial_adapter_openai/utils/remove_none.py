from typing import Any, Dict, Optional


def remove_none(d: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    """
    Remove keys with None values from a dictionary.

    :param d: The input dictionary with potential None values.
    :return: A new dictionary with keys containing None values removed.
    """
    return {k: v for k, v in d.items() if v is not None}
