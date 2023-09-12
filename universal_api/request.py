from enum import Enum
from typing import List, Literal, Optional

from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from llm.exception import ValidationError


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "function"]
    content: Optional[str] = None
    name: Optional[str] = None
    function_call: Optional[str] = None

    class Config:
        extra = "allow"

    def to_base_message(self) -> BaseMessage:
        assert self.content is not None
        match self.role:
            case "system":
                return SystemMessage(content=self.content)
            case "user":
                return HumanMessage(content=self.content)
            case "assistant":
                return AIMessage(content=self.content)
            case "function":
                raise ValidationError("Function calls aren't supported")


class CompletionParameters(BaseModel):
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    stop: Optional[List[str]] = None
    n: Optional[int] = None
    stream: Optional[bool] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None


# Direct translation of https://platform.openai.com/docs/api-reference/chat/create
class ChatCompletionQuery(CompletionParameters, BaseModel):
    messages: List[Message]

    class Config:
        extra = "allow"

    @staticmethod
    def example() -> "ChatCompletionQuery":
        return ChatCompletionQuery(
            max_tokens=1024,
            temperature=0.0,
            messages=[
                Message(role="user", content="Who are you?"),
            ],
        )


class CompletionQuery(CompletionParameters, BaseModel):
    prompt: str

    class Config:
        extra = "allow"


class EmbeddingsType(str, Enum):
    SYMMETRIC = "symmetric"
    DOCUMENT = "document"
    QUERY = "query"


class EmbeddingsQuery(BaseModel):
    input: str | List[str]
    user: Optional[str] = None

    @staticmethod
    def example() -> "EmbeddingsQuery":
        return EmbeddingsQuery(input=["fish", "ball"])
