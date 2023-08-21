from enum import Enum

from utils.cli import select_enum


class VertexAIModels(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"


def choose_model() -> VertexAIModels:
    return select_enum("Select the model", VertexAIModels)
