from typing import List, Optional, TypedDict, Union


class MessageData(TypedDict):
    role: str
    content: str


class ChatCompletionRequestData(TypedDict):
    model: str
    messages: List[MessageData]
    max_tokens: Optional[int]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]
    stream: Optional[bool]
    stop: Optional[Union[str, List[str]]]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
    logit_bias: Optional[dict]
    user: Optional[str]
