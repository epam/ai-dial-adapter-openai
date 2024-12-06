from typing import Any


def match_objects(expected: Any, actual: Any) -> None:
    if isinstance(expected, dict):
        assert list(sorted(expected.keys())) == list(sorted(actual.keys()))
        for k, v in expected.items():
            match_objects(v, actual[k])
    elif isinstance(expected, tuple):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            match_objects(expected[i], actual[i])
    elif isinstance(expected, list):
        assert len(expected) == len(actual)
        for i in range(len(expected)):
            match_objects(expected[i], actual[i])
    elif callable(expected):
        assert expected(actual)
    else:
        assert expected == actual
