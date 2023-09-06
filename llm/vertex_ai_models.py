from enum import Enum


class VertexAIModelName(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CODECHAT_BISON_1 = "codechat-bison@001"


class ExtraVertexAIModelName(str, Enum):
    DOLLY_V2_7B = "dolly-v2-7b"
    LLAMA2_7B_CHAT_1 = "llama2-7b-chat-001"
