import logging
import re
from typing import Any, TypeVar

_log = logging.getLogger(__name__)


def match_objects(expected: Any, actual: Any) -> bool:
    if isinstance(expected, dict):
        assert sorted(expected.keys()) == sorted(actual.keys())
        for k, v in expected.items():
            match_objects(v, actual[k])
    elif isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            match_objects(expected[i], actual[i])
    elif callable(expected):
        assert expected(actual), (
            f"The object doesn't satisfy test predicate: {actual}"
        )
    else:
        assert expected == actual

    return True


_T = TypeVar("_T")


def cleanup_repeated_tags(o: _T, path: str = "") -> _T:
    """
    Certain models like Grok return an invalid stream of chunks,
    whose merge results into an invalid Chat Completion response
    failing OpenAI SDK validation.
    We attempt to fix it in this helper.
    The input object is mutated inplace.
    """

    _path_patterns = [r".choices\.[0-9]+\.finish_reason"]

    def _remove_repetition(string: str) -> str:
        n = len(string)
        for i in range(2, n):
            if n % i == 0:
                m = n // i
                prefix = string[:i]
                if prefix * m == string:
                    _log.warning(
                        f"Model returned a repeated string value at {path!r}: {string!r}. "
                        f"Collapsing it to a single repetition: {prefix!r}."
                    )
                    return prefix
        return string

    if isinstance(o, str):
        for pattern in _path_patterns:
            if re.fullmatch(pattern, path):
                return _remove_repetition(o)

    if isinstance(o, list):
        for i, e in enumerate(o):
            o[i] = cleanup_repeated_tags(e, path + f".{i}")

    if isinstance(o, dict):
        for k, v in o.items():
            o[k] = cleanup_repeated_tags(v, path + f".{k}")

    return o
