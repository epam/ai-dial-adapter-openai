from typing import Literal, overload

from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.tokenizer import (
    MultiModalTokenizer,
    PlainTextTokenizer,
)


@overload
def get_tokenizer(
    tiktoken_model: str,
    deployment_type: Literal[
        ChatCompletionDeploymentType.GPT4_VISION,
        ChatCompletionDeploymentType.GPT4O,
        ChatCompletionDeploymentType.GPT4O_MINI,
    ],
) -> MultiModalTokenizer: ...


@overload
def get_tokenizer(
    tiktoken_model: str,
    deployment_type: Literal[ChatCompletionDeploymentType.GPT_TEXT_ONLY],
) -> PlainTextTokenizer: ...


@overload
def get_tokenizer(
    tiktoken_model: str,
    deployment_type: Literal[
        ChatCompletionDeploymentType.DALLE3,
        ChatCompletionDeploymentType.MISTRAL,
        ChatCompletionDeploymentType.DATABRICKS,
    ],
) -> None: ...


def get_tokenizer(
    tiktoken_model: str, deployment_type: ChatCompletionDeploymentType
) -> MultiModalTokenizer | PlainTextTokenizer | None:
    match deployment_type:
        case (
            ChatCompletionDeploymentType.GPT4_VISION
            | ChatCompletionDeploymentType.GPT4O
            | ChatCompletionDeploymentType.GPT4O_MINI
        ):
            return MultiModalTokenizer(
                tiktoken_model, get_image_tokenizer(deployment_type)
            )
        case ChatCompletionDeploymentType.GPT_TEXT_ONLY:
            return PlainTextTokenizer(model=tiktoken_model)
        case _:
            return None
