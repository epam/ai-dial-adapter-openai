from typing import Any, Iterable, Literal, Self

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
    def finish_reasons(self) -> Iterable[Any]:
        for choice in self.response.get("choices") or []:
            if (reason := choice.get("finish_reason")) is not None:
                yield reason

    @property
    def has_finish_reason(self) -> bool:
        return len(list(self.finish_reasons)) > 0

    @property
    def messages(self) -> Iterable[Any]:
        for choice in self.response.get("choices") or []:
            if (message := choice.get(self.message_key)) is not None:
                yield message

    @property
    def has_messages(self) -> bool:
        return len(list(self.messages)) > 0


class ChatCompletionBlock(ChatCompletionResponse):
    def __init__(self, response: dict):
        super().__init__(message_key="message", response=response)


class ChatCompletionStreamingChunk(ChatCompletionResponse):
    def __init__(self, response: dict):
        super().__init__(message_key="delta", response=response)

    def merge(self, chunk: dict) -> Self:
        self.response = merge_chat_completion_chunks(self.response, chunk)
        return self
