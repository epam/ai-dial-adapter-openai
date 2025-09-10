"""
Implemented based on the official recipe: https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
"""

import json
from abc import ABC, abstractmethod
from typing import Any, Callable, Coroutine, Generic, List, Set, TypeVar

from aidial_sdk.exceptions import InternalServerError
from tiktoken import Encoding, encoding_for_model
from tiktoken.model import MODEL_PREFIX_TO_ENCODING

from aidial_adapter_openai.utils.chat_completion_response import (
    ChatCompletionResponse,
)
from aidial_adapter_openai.utils.concurrency import run_in_threadpool
from aidial_adapter_openai.utils.image_tokenizer import (
    IMAGE_SUPPORTING_DEPLOYMENTS,
    ImageTokenizer,
)
from aidial_adapter_openai.utils.log_config import logger
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


class BaseTokenizer(ABC, Generic[MessageType]):
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

    async def tokenize_text(self, text: str) -> int:
        return await run_in_threadpool(lambda: len(self.encoding.encode(text)))

    async def tokenize_response(self, resp: ChatCompletionResponse) -> int:
        return sum(
            [
                await self._tokenize_response_message(message)
                for message in resp.messages
            ]
        )

    async def _tokenize_object(self, obj: Any) -> int:
        if not obj:
            return 0

        # OpenAI doesn't reveal tokenization algorithm for tools calls and function calls.
        # An approximation is used instead - token count in the string repr of the objects.
        text = (
            obj
            if isinstance(obj, str)
            else json.dumps(obj, separators=(",", ":"))
        )
        return await self.tokenize_text(text)

    async def _tokenize_response_message(self, message: dict) -> int:
        tokens = 0

        for key in ["content", "refusal", "function"]:
            tokens += await self._tokenize_object(message.get(key))

        for tool_call in message.get("tool_calls") or []:
            tokens += await self._tokenize_object(tool_call.get("function"))

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

    async def tokenize_request(
        self, original_request: dict, messages: List[MessageType]
    ) -> int:
        tokens = self.TOKENS_PER_REQUEST

        if original_request.get("function_call") != "none":
            for func in original_request.get("function") or []:
                tokens += await self._tokenize_object(func)

        if original_request.get("tool_choice") != "none":
            for tool in original_request.get("tools") or []:
                tokens += await self._tokenize_object(tool.get("function"))

        tokens += sum(
            [
                await self.tokenize_request_message(message)
                for message in messages
            ]
        )

        return tokens

    @abstractmethod
    async def tokenize_request_message(self, message: MessageType) -> int:
        pass


async def _tokenize_message(
    message: dict,
    tokens_per_name: int,
    tokenize_text: Callable[[str], Coroutine[None, None, int]],
    tokenize_multi_modal_content_part: Callable[
        [Any], Coroutine[None, None, int]
    ],
) -> int:
    tokens = 0
    for key, value in message.items():
        if key == "name":
            tokens += tokens_per_name

        elif key == "content":
            match value:
                case None:
                    pass
                case list():
                    for content_part in value:
                        if content_part["type"] == "text":
                            tokens += await tokenize_text(content_part["text"])
                        else:
                            tokens += await tokenize_multi_modal_content_part(
                                content_part
                            )
                case str():
                    tokens += await tokenize_text(value)
                case _:
                    raise InternalServerError(
                        f"Unexpected type of content in message: {type(value)}"
                    )

        elif key == "role":
            if isinstance(value, str):
                tokens += await tokenize_text(value)
            else:
                raise InternalServerError(
                    f"Unexpected type of 'role' field in message: {type(value)}"
                )
    return tokens


class Tokenizer(BaseTokenizer[MultiModalMessage]):
    image_tokenizer: ImageTokenizer | None
    warnings: Set[str]

    def __init__(
        self, *, model: str, image_tokenizer: ImageTokenizer | None = None
    ):
        super().__init__(model)
        self.image_tokenizer = image_tokenizer
        self.warnings = set()

    async def _on_multi_modal_content_part(self, content_part: dict) -> int:
        ty = content_part.get("type")
        if ty == "image_url" and self.image_tokenizer is None:
            env_vars = " or ".join(IMAGE_SUPPORTING_DEPLOYMENTS)
            self.warnings.add(
                "Image content detected, however, the image tokenization algorithm is not known for this deployment. "
                "Tokens for the image will be ignored. "
                f"Declare the deployment in either {env_vars} to specify the image tokenization algorithm."
            )

        if ty != "image_url":
            self.warnings.add(
                f"Content part type {ty!r} is not supported by the tokenizer. "
                "Tokens for this content part will be ignored."
            )

        return 0

    async def tokenize_request_message(self, message: MultiModalMessage) -> int:
        tokens = self._tokens_per_request_message

        tokens += await _tokenize_message(
            message=message.raw_message,
            tokens_per_name=self._tokens_per_request_message_name,
            tokenize_text=self.tokenize_text,
            tokenize_multi_modal_content_part=self._on_multi_modal_content_part,
        )

        # Processing image parts of message
        for metadata in message.images:
            if self.image_tokenizer is not None:
                tokens += self.image_tokenizer.tokenize(
                    width=metadata.width,
                    height=metadata.height,
                    detail=metadata.detail,
                )

        return tokens

    async def tokenize_request(
        self, original_request: dict, messages: List[MultiModalMessage]
    ) -> int:
        tokens = await super().tokenize_request(original_request, messages)

        for warning in self.warnings:
            logger.warning(warning)
        self.warnings.clear()

        return tokens
