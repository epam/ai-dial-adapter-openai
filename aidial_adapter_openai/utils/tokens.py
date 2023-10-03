"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""
from typing import Any, List

from tiktoken import Encoding


def calculate_prompt_tokens(
    messages: List[Any], model: str, encoding: Encoding
):
    prompt_tokens = 3

    if model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4
        tokens_per_name = -1
    else:
        tokens_per_message = 3
        tokens_per_name = 1

    for message in messages:
        prompt_tokens += tokens_per_message

        for key, value in message.items():
            prompt_tokens += len(encoding.encode(value))
            if key == "name":
                prompt_tokens += tokens_per_name

    return prompt_tokens
