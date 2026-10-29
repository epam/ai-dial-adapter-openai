from openai.types.shared_params import Reasoning, ReasoningEffort
from pydantic import BaseModel, Field

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.validation import RequestValue


class ResponsesConfig(ExtraAllowedModel):
    reasoning: Reasoning | None = Field(
        default=None,
        description="Configuration options for [reasoning models](https://platform.openai.com/docs/guides/reasoning).",
    )


class _ChatCompletionReasoning(BaseModel):
    reasoning_effort: ReasoningEffort = None


def get_configuration(request: RequestValue) -> ResponsesConfig:
    configuration = (
        parse_configuration(ResponsesConfig, request) or ResponsesConfig()
    )

    if configuration.reasoning is None and (
        effort := request.to_model(_ChatCompletionReasoning).reasoning_effort
    ):
        configuration.reasoning = Reasoning(effort=effort)

    logger.debug(f"configuration: {configuration.model_dump_json()}")
    return configuration
