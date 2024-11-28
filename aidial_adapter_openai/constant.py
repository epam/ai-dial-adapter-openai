from enum import StrEnum, auto


class ChatCompletionDeploymentType(StrEnum):
    DALLE3 = auto()
    MISTRAL = auto()
    DATABRICKS = auto()
    GPT4_VISION = auto()
    GPT4O = auto()
    GPT4O_MINI = auto()
    GPT_TEXT_ONLY = auto()
