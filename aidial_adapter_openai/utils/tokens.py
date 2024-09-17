"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

from typing import Any, List

from aidial_sdk.exceptions import InternalServerError
from tiktoken import Encoding, encoding_for_model


class Tokenizer:
    """
    Tokenizer for message.
    Calculates only textual tokens, not image tokens.
    """

    model: str
    encoding: Encoding

    # Tokens that are added to all message tokens, not matter what
    PROMPT_TOKENS = 3

    def __init__(self, model: str) -> None:
        self.model = model
        try:
            self.encoding = encoding_for_model(model)
        except KeyError as e:
            raise InternalServerError(
                f"Could not find tokenizer for the model {model!r} in tiktoken. "
                "Consider mapping the model to an existing tokenizer via MODEL_ALIASES env var, "
                "or declare it as a model which doesn't require tokenization through tiktoken.",
            ) from e

    def calculate_tokens(self, string: str) -> int:
        return len(self.encoding.encode(string))

    def calculate_tokens_per_message(self, message: dict) -> int:
        if self.model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4
            tokens_per_name = -1
        else:
            tokens_per_message = 3
            tokens_per_name = 1

        tokens = tokens_per_message
        for key, value in message.items():
            if key == "name":
                tokens += tokens_per_name
            elif isinstance(value, str):
                tokens += self.calculate_tokens(value)
            if key == "content" and isinstance(value, list):
                for submessage in value:
                    if submessage["type"] == "text":
                        tokens += self.calculate_tokens(submessage["text"])
        return tokens

    def calculate_overall_prompt_tokens(self, message_tokens: int):
        return self.PROMPT_TOKENS + message_tokens

    def calculate_prompt_tokens(self, messages: List[Any]) -> int:
        return self.calculate_overall_prompt_tokens(
            message_tokens=sum(map(self.calculate_tokens_per_message, messages))
        )

    def available_message_tokens(self, max_prompt_tokens: int):
        return max_prompt_tokens - self.PROMPT_TOKENS
