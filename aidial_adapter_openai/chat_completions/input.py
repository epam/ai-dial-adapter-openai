from functools import cache
from typing import List, Literal, assert_never

from pydantic import BaseModel

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)


class SupportedInputs(BaseModel):
    input_types: List[str] | None = None
    usage_message: str | None = None


def image_inputs_supported() -> SupportedInputs:
    # Officially supported image types by GPT-4o and o-series models
    image_exts = ["jpg", "jpeg", "png", "webp", "gif"]
    image_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]

    usage = f"""
The model answers queries about attached images.
Attach images and ask questions about them.

Supported image types: {', '.join(image_exts)}.

Examples of queries:
- "Describe this picture" for one image,
- "What are in these images? Is there any difference between them?" for multiple images.
""".strip()

    return SupportedInputs(input_types=image_types, usage_message=usage)


def all_inputs_supported() -> SupportedInputs:
    return SupportedInputs()


GPTDeployment = Literal[
    ChatCompletionDeploymentType.GPT4O,
    ChatCompletionDeploymentType.GPT4O_MINI,
    ChatCompletionDeploymentType.GPT_TEXT_ONLY,
    ChatCompletionDeploymentType.RESPONSES_API,
]


@cache
def get_supported_inputs(deployment_type: GPTDeployment) -> SupportedInputs:
    match deployment_type:
        case (
            ChatCompletionDeploymentType.GPT4O
            | ChatCompletionDeploymentType.GPT4O_MINI
            | ChatCompletionDeploymentType.RESPONSES_API
        ):
            return image_inputs_supported()
        case ChatCompletionDeploymentType.GPT_TEXT_ONLY:
            return all_inputs_supported()
        case _:
            assert_never(deployment_type)
