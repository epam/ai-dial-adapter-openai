from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Literal, TypeVar, assert_never

from openai import NotGiven
from pydantic import BaseModel

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)

_Config = TypeVar("_Config", bound=BaseModel)


class ImageGenerationModel(ABC, Generic[_Config]):
    @abstractmethod
    def get_azure_api_version(self, config: ApplicationConfig) -> str: ...

    @abstractmethod
    def get_configuration(self) -> type[_Config]: ...

    @abstractmethod
    def get_response_format(self) -> NotGiven | Literal["b64_json"]: ...

    @abstractmethod
    def get_image_content_type(self, config: _Config) -> str: ...

    @classmethod
    def create(
        cls,
        deployment_type: Literal[
            ChatCompletionDeploymentType.GPT_IMAGE_1,
            ChatCompletionDeploymentType.DALLE3,
        ],
    ) -> ImageGenerationModel:
        match deployment_type:
            case ChatCompletionDeploymentType.DALLE3:
                from .dalle import Dalle3Model

                return Dalle3Model()
            case ChatCompletionDeploymentType.GPT_IMAGE_1:
                from .gpt_image import GptImage1Model

                return GptImage1Model()
            case _:
                assert_never(deployment_type)
