"""Generic token counter for models which do not provide a token counter."""
from langchain.base_language import _get_token_ids_default_method


def get_num_tokens(text: str) -> int:
    return len(_get_token_ids_default_method(text))
