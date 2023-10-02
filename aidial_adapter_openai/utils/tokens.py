from typing import Any, List

from tiktoken import Encoding


def calculate_prompt_tokens(
    messages: List[Any], model: str, encoding: Encoding
):
    prompt_tokens = 3
    tokens_per_message = (
        4 if model == "gpt-3.5-turbo-0301" else 3
    )  # possible need change gpt-3.5-turbo to something anything

    for message in messages:
        prompt_tokens += tokens_per_message

        for key, value in message.items():
            prompt_tokens += len(encoding.encode(value))
            if key == "name":
                prompt_tokens += 1

    return prompt_tokens
