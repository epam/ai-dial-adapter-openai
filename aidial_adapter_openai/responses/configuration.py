from typing import Any

from openai.types.shared_params import Reasoning
from pydantic import Field

from aidial_adapter_openai.dial_api.request import parse_configuration
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class ResponsesConfig(ExtraAllowedModel):
    reasoning: Reasoning | None = Field(
        default=None,
        description="Configuration options for [reasoning models](https://platform.openai.com/docs/guides/reasoning).",
    )


def get_configuration(request: dict[str, Any]) -> ResponsesConfig:
    configuration = (
        parse_configuration(ResponsesConfig, request) or ResponsesConfig()
    )

    if configuration.reasoning is None and (
        reasoning_effort := request.get("reasoning_effort")
    ):
        configuration.reasoning = Reasoning(effort=reasoning_effort)

    logger.debug(f"configuration: {configuration.model_dump_json()}")
    return configuration
