from enum import StrEnum


class ChatCompletionDeploymentType(StrEnum):
    GPT_IMAGE_1 = "GPT_IMAGE_1"
    DALLE3 = "DALLE3"
    MISTRAL = "MISTRAL"
    DATABRICKS = "DATABRICKS"
    GPT4O = "GPT4O"
    GPT4O_MINI = "GPT4O_MINI"
    GPT_TEXT_ONLY = "GPT_TEXT_ONLY"
