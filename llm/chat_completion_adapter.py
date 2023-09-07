from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from langchain.schema import AIMessage, BaseMessage, SystemMessage
from vertexai.language_models._language_models import ChatMessage

from universal_api.token_usage import TokenUsage


def to_chat_message(message: BaseMessage) -> ChatMessage:
    author = "bot" if isinstance(message, AIMessage) else "user"
    return ChatMessage(author=author, content=message.content)


class ChatCompletionAdapter(ABC):
    @abstractmethod
    async def _call(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[str, TokenUsage]:
        pass

    async def completion(self, prompt: str) -> Tuple[str, TokenUsage]:
        return await self._call(None, [], prompt)

    async def chat(self, history: List[BaseMessage]) -> Tuple[str, TokenUsage]:
        messages = history.copy()

        context: Optional[str] = None
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            context = messages.pop(0).content

        if len(messages) == 0:
            raise Exception(
                "The chat message must have at least one message besides initial system message"
            )

        message_history = list(map(to_chat_message, messages[:-1]))

        return await self._call(context, message_history, messages[-1].content)
