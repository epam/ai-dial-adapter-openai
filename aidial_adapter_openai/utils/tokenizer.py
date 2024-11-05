"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

from abc import abstractmethod
from typing import Any, Callable, Generic, List, TypeVar

from aidial_sdk.exceptions import InternalServerError
from tiktoken import Encoding, encoding_for_model

from aidial_adapter_openai.utils.image_tokenizer import ImageTokenizer
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage

MessageType = TypeVar("MessageType")


class BaseTokenizer(Generic[MessageType]):
    model: str
    encoding: Encoding
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

    def calculate_text_tokens(self, text: str) -> int:
        return len(self.encoding.encode(text))

    @property
    def tokens_per_message(self) -> int:
        """
        Tokens, that are counter for each message, regardless of its content
        """
        if self.model == "gpt-3.5-turbo-0301":
            return 4
        return 3

    @property
    def tokens_per_name(self) -> int:
        """
        Tokens, that are counter for "name" field in message, if it's present
        """
        if self.model == "gpt-3.5-turbo-0301":
            return -1
        return 1

    def calculate_request_prompt_tokens(self, messages_tokens: int):
        """
        Amount of tokens, that will be counted by API
        is greater than actual sum of tokens of all messages
        """
        return self.TOKENS_PER_REQUEST + messages_tokens

    def calculate_prompt_tokens(self, messages: List[MessageType]) -> int:
        return self.calculate_request_prompt_tokens(
            messages_tokens=sum(map(self.calculate_message_tokens, messages))
        )

    def available_message_tokens(self, max_prompt_tokens: int):
        return max_prompt_tokens - self.TOKENS_PER_REQUEST

    @abstractmethod
    def calculate_message_tokens(self, message: MessageType) -> int:
        pass


def _process_raw_message(
    raw_message: dict,
    tokens_per_name: int,
    calculate_text_tokens: Callable[[str], int],
    handle_custom_content_part: Callable[[Any], None],
) -> int:
    tokens = 0
    for key, value in raw_message.items():
        if key == "name":
            tokens += tokens_per_name

        elif key == "content":
            if isinstance(value, list):
                for content_part in value:
                    if content_part["type"] == "text":
                        tokens += calculate_text_tokens(content_part["text"])
                    else:
                        handle_custom_content_part(content_part)

            elif isinstance(value, str):
                tokens += calculate_text_tokens(value)
            elif value is None:
                pass
            else:
                raise InternalServerError(
                    f"Unexpected type of content in message: {value!r}"
                )

        elif key == "role":
            if isinstance(value, str):
                tokens += calculate_text_tokens(value)
            else:
                raise InternalServerError(
                    f"Unexpected type of 'role' field in message: {value!r}"
                )
    return tokens


class PlainTextTokenizer(BaseTokenizer[dict]):
    """
    Tokenizer for message.
    Calculates only textual tokens, not image tokens.
    """

    def _handle_custom_content_part(self, content_part: Any):
        raise InternalServerError(
            f"Unexpected type of content in message: {content_part!r}"
            f"Use MultiModalTokenizer for messages with images"
        )

    def calculate_message_tokens(self, message: dict) -> int:
        return self.tokens_per_message + _process_raw_message(
            raw_message=message,
            tokens_per_name=self.tokens_per_name,
            calculate_text_tokens=self.calculate_text_tokens,
            handle_custom_content_part=self._handle_custom_content_part,
        )


class MultiModalTokenizer(BaseTokenizer[MultiModalMessage]):
    image_tokenizer: ImageTokenizer

    def __init__(self, model: str, image_tokenizer: ImageTokenizer):
        super().__init__(model)
        self.image_tokenizer = image_tokenizer

    def calculate_message_tokens(self, message: MultiModalMessage) -> int:
        tokens = self.tokens_per_message
        raw_message = message.raw_message

        tokens += _process_raw_message(
            raw_message=raw_message,
            tokens_per_name=self.tokens_per_name,
            calculate_text_tokens=self.calculate_text_tokens,
            handle_custom_content_part=lambda content_part: None,
        )

        # Processing image parts of message
        for metadata in message.image_metadatas:
            tokens += self.image_tokenizer.tokenize(
                width=metadata.width,
                height=metadata.height,
                detail=metadata.detail,
            )
        return tokens
