from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import AIMessage, BaseMessage, SystemMessage
from vertexai.language_models._language_models import ChatMessage

from llm.exception import ValidationError
from universal_api.token_usage import TokenUsage


def to_chat_message(message: BaseMessage) -> ChatMessage:
    author = "bot" if isinstance(message, AIMessage) else "user"
    return ChatMessage(author=author, content=message.content)


ChatCompletionResponse = Tuple[str, TokenUsage]


class ChatCompletionAdapter(ABC):
    @abstractmethod
    async def _call(
        self,
        streaming: bool,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> ChatCompletionResponse:
        pass

    async def completion(
        self, streaming: bool, prompt: str
    ) -> ChatCompletionResponse:
        return await self._call(streaming, None, [], prompt)

    async def chat(
        self, streaming: bool, history: List[BaseMessage]
    ) -> ChatCompletionResponse:
        messages = history.copy()

        context: Optional[str] = None
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            context = messages.pop(0).content

        if len(messages) == 0:
            raise ValidationError(
                "The chat message must have at least one message besides initial system message"
            )

        message_history = list(map(to_chat_message, messages[:-1]))

        return await self._call(
            streaming, context, message_history, messages[-1].content
        )
