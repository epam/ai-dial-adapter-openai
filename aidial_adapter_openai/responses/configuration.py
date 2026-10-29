from openai.types.shared_params import Reasoning
from pydantic import Field

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.validation import RequestValue


class ResponsesConfig(ExtraAllowedModel):
    reasoning: Reasoning | None = Field(
        default=None,
        description="Configuration options for [reasoning models](https://platform.openai.com/docs/guides/reasoning).",
    )


def get_configuration(request: RequestValue) -> ResponsesConfig:
    configuration = (
        parse_configuration(ResponsesConfig, request) or ResponsesConfig()
    )

    if configuration.reasoning is None and (
        reasoning_effort := request.get("reasoning_effort")
    ):
        # FIX_NOW: here we need to throw an invalid request exception if
        # reasoning_effort.unwrap() cannot be parsed into Reasoning.effort.
        # Currently, it would read to just runtime error, I believe.
        configuration.reasoning = Reasoning(effort=reasoning_effort.unwrap())

    logger.debug(f"configuration: {configuration.model_dump_json()}")
    return configuration
