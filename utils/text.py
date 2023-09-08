import re
from typing import List, Optional


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


# Copy of langchain.llms.utils::enforce_stop_tokens with a bugfix: stop words are escaped.
def enforce_stop_tokens(text: str, stop: Optional[List[str]]) -> str:
    """Cut off the text as soon as any stop words occur."""

    if stop is None:
        return text

    stop_escaped = [re.escape(s) for s in stop]
    return re.split("|".join(stop_escaped), text)[0]
