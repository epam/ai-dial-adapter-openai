from typing import Any, Iterable, Literal, Self

from aidial_sdk.utils.merge_chunks import merge_chat_completion_chunks
from pydantic import BaseModel


class ChatCompletionResponse(BaseModel):
    message_key: Literal["delta", "message"]
    resp: dict = {}

    @property
    def usage(self) -> Any | None:
        return self.resp.get("usage")

    @property
    def is_empty(self) -> bool:
        return self.resp == {}

    @property
    def finish_reasons(self) -> Iterable[Any]:
        for choice in self.resp.get("choices") or []:
            if (reason := choice.get("finish_reason")) is not None:
                yield reason

    @property
    def has_finish_reason(self) -> bool:
        return len(list(self.finish_reasons)) > 0

    @property
    def messages(self) -> Iterable[Any]:
        for choice in self.resp.get("choices") or []:
            if (message := choice.get(self.message_key)) is not None:
                yield message

    @property
    def has_messages(self) -> bool:
        return len(list(self.messages)) > 0


class ChatCompletionBlock(ChatCompletionResponse):
    def __init__(self, **kwargs):
        super().__init__(message_key="message", **kwargs)


class ChatCompletionStreamingChunk(ChatCompletionResponse):
    def __init__(self, **kwargs):
        super().__init__(message_key="delta", **kwargs)

    def merge(self, chunk: dict) -> Self:
        self.resp = merge_chat_completion_chunks(self.resp, chunk)
        return self
