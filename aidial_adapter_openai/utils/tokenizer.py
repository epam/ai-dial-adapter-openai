"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

from typing import Any, List

from aidial_sdk.exceptions import InternalServerError
from tiktoken import Encoding, encoding_for_model

from aidial_adapter_openai.utils.image_tokenizer import tokenize_image_data_url


class Tokenizer:
    """
    Tokenizer for message.
    Calculates only textual tokens, not image tokens.
    """

    model: str
    encoding: Encoding

    # Tokens that are added to all message tokens, not matter what
    TOKENS_PER_REQUEST = 3

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

    @property
    def tokens_per_message(self) -> int:
        if self.model == "gpt-3.5-turbo-0301":
            return 4
        return 3

    @property
    def tokens_per_name(self) -> int:
        if self.model == "gpt-3.5-turbo-0301":
            return -1
        return 1

    def calculate_tokens_per_message(self, message: dict) -> int:
        tokens = self.tokens_per_message
        for key, value in message.items():
            if key == "name":
                tokens += self.tokens_per_name
            elif key == "content" and isinstance(value, list):
                for submessage in value:
                    if submessage["type"] == "text":
                        tokens += self.calculate_tokens(submessage["text"])
            elif isinstance(value, str):
                tokens += self.calculate_tokens(value)
            elif key == "content" and not isinstance(value, str):
                raise InternalServerError(
                    f"Unexpected type of content in message: {value!r}"
                    f"Use MultiModalTokenizer for messages with images"
                )

        return tokens

    def calculate_request_prompt_tokens(self, messages_tokens: int):
        """
        Amount of tokens, that will be counted by API
        is greater than actual sum of tokens of all messages by PROMPT_TOKENS
        """
        return self.TOKENS_PER_REQUEST + messages_tokens

    def calculate_prompt_tokens(self, messages: List[Any]) -> int:
        return self.calculate_request_prompt_tokens(
            messages_tokens=sum(
                map(self.calculate_tokens_per_message, messages)
            )
        )

    def available_message_tokens(self, max_prompt_tokens: int):
        return max_prompt_tokens - self.TOKENS_PER_REQUEST


class MutliModalTokenizer(Tokenizer):
    def calculate_tokens_per_message(self, message: dict) -> int:
        tokens = self.tokens_per_message
        for key, value in message.items():
            if key == "name":
                tokens += self.tokens_per_name
            elif key == "content" and isinstance(value, list):
                for submessage in value:
                    # Process text part of the message
                    if submessage["type"] == "text":
                        tokens += self.calculate_tokens(submessage["text"])
                    # Process image part of the message
                    elif submessage["type"] == "image_url":
                        image_url = submessage["image_url"]
                        tokens += tokenize_image_data_url(
                            image_url["url"], image_url["detail"]
                        )
            elif isinstance(value, str):
                tokens += self.calculate_tokens(value)
        return tokens
