from enum import StrEnum


class ChatCompletionDeploymentType(StrEnum):
    DALLE3 = "DALLE3"
    MISTRAL = "MISTRAL"
    DATABRICKS = "DATABRICKS"
    GPT4_VISION = "GPT4_VISION"
    GPT4O = "GPT4O"
    GPT4O_MINI = "GPT4O_MINI"
    GPT_TEXT_ONLY = "GPT_TEXT_ONLY"
