import re

from rapidfuzz.distance import Levenshtein


def is_close_enough(expected: str, actual: str, limit: int = 4) -> bool:
    def _sanitize_text(s: str) -> str:
        ret = re.sub("[^a-z]", " ", s.lower())
        ret = re.sub("( )+", " ", ret)
        return ret.strip()

    expected, actual = _sanitize_text(expected), _sanitize_text(actual)

    dist = Levenshtein.distance(expected, actual)
    assert (
        dist <= limit
    ), f"Levenshtein distance between {expected!r} and {actual!r} is too big: {dist}, but expected under {limit}"

    return True
