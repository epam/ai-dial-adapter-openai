"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

import json
from abc import abstractmethod
from typing import Any, Callable, Generic, List, NoReturn, TypeVar

from aidial_sdk.exceptions import InternalServerError
from tiktoken import Encoding, encoding_for_model
from tiktoken.model import MODEL_PREFIX_TO_ENCODING

from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionResponse,
)
from aidial_adapter_openai.utils.image_tokenizer import ImageTokenizer
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage

MessageType = TypeVar("MessageType")


_TIKTOKEN_MODEL_PREFIXES = [
    f'"{p}"' for p in MODEL_PREFIX_TO_ENCODING.keys() if not p.startswith("ft:")
]


def _get_tiktoken_error_message(model: str) -> str:
    var_name = "TIKTOKEN_MODEL_MAPPING"

    return (
        f"Could not find tokenizer for the model {model!r} in the tiktoken package. "
        f"Consider mapping the model to an existing tokenizer via {var_name} variable in the adapter OpenAI environment: "
        f'{var_name}=\'{{"{model}": $prefix}}\', where $prefix is one of: {", ".join(_TIKTOKEN_MODEL_PREFIXES)}. '
        "Alternatively, declare the deployment as a model that doesn't require tokenization via tiktoken."
    )


class BaseTokenizer(Generic[MessageType]):
    """
    Tokenizer for chat completion requests and responses.
    """

    model: str
    encoding: Encoding
    TOKENS_PER_REQUEST = 3

    def __init__(self, model: str) -> None:
        self.model = model
        try:
            self.encoding = encoding_for_model(model)
        except KeyError as e:
            raise InternalServerError(_get_tiktoken_error_message(model)) from e

    def tokenize_text(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def tokenize_response(self, resp: ChatCompletionResponse) -> int:
        return sum(map(self._tokenize_response_message, resp.messages))

    def _tokenize_object(self, obj: Any) -> int:
        if not obj:
            return 0

        # OpenAI doesn't reveal tokenization algorithm for tools calls and function calls.
        # An approximation is used instead - token count in the string repr of the objects.
        text = (
            obj
            if isinstance(obj, str)
            else json.dumps(obj, separators=(",", ":"))
        )
        return self.tokenize_text(text)

    def _tokenize_response_message(self, message: dict) -> int:
        tokens = 0

        for key in ["content", "refusal", "function"]:
            tokens += self._tokenize_object(message.get(key))

        for tool_call in message.get("tool_calls") or []:
            tokens += self._tokenize_object(tool_call.get("function"))

        return tokens

    @property
    def _tokens_per_request_message(self) -> int:
        """
        Tokens, that are counter for each message, regardless of its content
        """
        if self.model == "gpt-3.5-turbo-0301":
            return 4
        return 3

    @property
    def _tokens_per_request_message_name(self) -> int:
        """
        Tokens, that are counter for "name" field in message, if it's present
        """
        if self.model == "gpt-3.5-turbo-0301":
            return -1
        return 1

    def tokenize_request(
        self, original_request: dict, messages: List[MessageType]
    ) -> int:
        tokens = self.TOKENS_PER_REQUEST

        if original_request.get("function_call") != "none":
            for func in original_request.get("function") or []:
                tokens += self._tokenize_object(func)

        if original_request.get("tool_choice") != "none":
            for tool in original_request.get("tools") or []:
                tokens += self._tokenize_object(tool.get("function"))

        tokens += sum(map(self.tokenize_request_message, messages))

        return tokens

    @abstractmethod
    def tokenize_request_message(self, message: MessageType) -> int:
        pass


def _tokenize_raw_message(
    raw_message: dict,
    tokens_per_name: int,
    tokenize_text: Callable[[str], int],
    tokenize_multi_modal_content_part: Callable[[Any], int],
) -> int:
    tokens = 0
    for key, value in raw_message.items():
        if key == "name":
            tokens += tokens_per_name

        elif key == "content":
            if isinstance(value, list):
                for content_part in value:
                    if content_part["type"] == "text":
                        tokens += tokenize_text(content_part["text"])
                    else:
                        tokens += tokenize_multi_modal_content_part(
                            content_part
                        )

            elif isinstance(value, str):
                tokens += tokenize_text(value)
            elif value is None:
                pass
            else:
                raise InternalServerError(
                    f"Unexpected type of content in message: {type(value)}"
                )

        elif key == "role":
            if isinstance(value, str):
                tokens += tokenize_text(value)
            else:
                raise InternalServerError(
                    f"Unexpected type of 'role' field in message: {type(value)}"
                )
    return tokens


class Tokenizer(BaseTokenizer[MultiModalMessage]):
    image_tokenizer: ImageTokenizer | None

    def __init__(
        self, *, model: str, image_tokenizer: ImageTokenizer | None = None
    ):
        super().__init__(model)
        self.image_tokenizer = image_tokenizer

    def _reject_image(self) -> NoReturn:
        raise InternalServerError(
            "Unexpected message with an image. "
            "The deployment only supports plain text messages. "
            "Remove the image from the request or declare the deployment as a multi-modal one "
            "in the OpenAI adapter configuration to avoid the error."
        )

    def _accept_image_content_part(self, content_part: dict) -> int:
        if (ty := content_part.get("type")) == "image_url":
            if self.image_tokenizer is None:
                self._reject_image()
            return 0

        raise InternalServerError(
            f"Unsupported multi-modal content part of type {ty!r}."
        )

    def tokenize_request_message(self, message: MultiModalMessage) -> int:
        tokens = self._tokens_per_request_message

        tokens += _tokenize_raw_message(
            raw_message=message.raw_message,
            tokens_per_name=self._tokens_per_request_message_name,
            tokenize_text=self.tokenize_text,
            tokenize_multi_modal_content_part=self._accept_image_content_part,
        )

        # Processing image parts of the message
        for metadata in message.image_metadatas:
            if self.image_tokenizer is None:
                self._reject_image()

            tokens += self.image_tokenizer.tokenize(
                width=metadata.width,
                height=metadata.height,
                detail=metadata.detail,
            )

        return tokens
