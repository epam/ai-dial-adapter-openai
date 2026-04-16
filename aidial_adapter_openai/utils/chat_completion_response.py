from collections.abc import Iterable
from typing import Any, Literal, Self

from aidial_sdk.utils.merge_chunks import merge_chat_completion_chunks
from pydantic import BaseModel


class ChatCompletionResponse(BaseModel):
    message_key: Literal["delta", "message"]
    response: dict = {}

    @property
    def usage(self) -> Any | None:
        return self.response.get("usage")

    @property
    def is_empty(self) -> bool:
        return self.response == {}

    @property
    def messages(self) -> Iterable[Any]:
        for choice in self.response.get("choices") or []:
            if (message := choice.get(self.message_key)) is not None:
                yield message

    @property
    def has_messages(self) -> bool:
        return len(list(self.messages)) > 0

    def get_missing_finish_reasons(self, n: int) -> set[int]:
        missing = set(range(n))
        for choice in self.response.get("choices") or []:
            if choice.get("finish_reason") is not None:
                index = choice.get("index")
                missing.discard(index)
        return missing


class ChatCompletionBlock(ChatCompletionResponse):
    def __init__(self, response: dict):
        super().__init__(message_key="message", response=response)


class ChatCompletionStreamingChunk(ChatCompletionResponse):
    def __init__(self, response: dict):
        super().__init__(message_key="delta", response=response)

    def merge(self, chunk: dict) -> Self:
        self.response = merge_chat_completion_chunks(self.response, chunk)
        return self
