from __future__ import annotations

import functools
import json
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Literal,
    TypeVar,
    assert_never,
)

from pydantic import BaseModel

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel


class UpstreamConfig(ExtraAllowedModel):
    endpoint: str
    key: str | None


class Features(ExtraAllowedModel):
    """
    What features/parameters the model supports.
    * Function calling support: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/function-calling#function-calling-support
    * Reasoning support: https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/reasoning?tabs=python-secure%2Cpy#api--feature-support
    """

    systemPromptSupported: bool = True
    toolsSupported: bool = True
    parallelToolCallsSupported: bool = True
    temperatureSupported: bool = True

    # Not in the DIAL Core config yet
    maxTokensSupported: bool = True
    reasoningSupported: bool = False
    reasoningSummarySupported: bool = False
    stopSupported: bool = True
    imageGenerationSupported: bool = False
    imageEditingSupported: bool = False
    emptyDialogSupported: bool = True

    responseFormatJsonObjectSupported: bool = True
    responseFormatJsonSchemaSupported: bool = True


class ModelConfig(ExtraAllowedModel):
    type: Literal["chat", "embedding"]
    overrideName: str | None = None
    upstreams: List[UpstreamConfig]
    features: Features = Features()
    inputAttachmentTypes: List[str] | None = None


class CoreConfig(ExtraAllowedModel):
    models: Dict[str, ModelConfig]

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            test_config = json.load(f)

        return cls(**test_config)


class EmbeddingsDeploymentType(BaseModel):
    pass


_T = TypeVar("_T")

DeploymentType = ChatCompletionDeploymentType | EmbeddingsDeploymentType


def _classify_deployment(
    app_config: ApplicationConfig,
    model_config: ModelConfig,
    deployment_id: str,
    upstream_endpoint: str,
) -> DeploymentType:
    if model_config.type == "chat":
        return app_config.get_chat_completion_deployment_type(
            deployment_id, upstream_endpoint
        ).deployment_type
    elif model_config.type == "embedding":
        return EmbeddingsDeploymentType()
    else:
        assert_never(model_config.type)


class DeploymentConfig(BaseModel, Generic[_T]):
    id_: str
    type_: _T

    model_name: str
    model_features: Features
    model_attachments: List[str] | None

    upstream_endpoint: str
    upstream_api_key: str | None
    upstream_idx: int | None

    @property
    def upstream_headers(self) -> Dict[str, str]:
        headers = {"X-UPSTREAM-ENDPOINT": self.upstream_endpoint}
        if self.upstream_api_key is not None:
            headers["X-UPSTREAM-KEY"] = self.upstream_api_key
        return headers

    @classmethod
    def create_deployments(
        cls,
        core_config: CoreConfig,
        get_deployment_type: Callable[[ModelConfig, str, str], _T],
    ) -> List[DeploymentConfig[_T]]:
        configs = []
        for deployment_id, model_config in core_config.models.items():
            for upstream_index, upstream_config in enumerate(
                model_config.upstreams
            ):
                upstream_endpoint = upstream_config.endpoint

                upstream_idx = (
                    None if len(model_config.upstreams) <= 1 else upstream_index
                )

                configs.append(
                    cls(
                        upstream_idx=upstream_idx,
                        id_=deployment_id,
                        model_name=model_config.overrideName or deployment_id,
                        model_features=model_config.features,
                        model_attachments=model_config.inputAttachmentTypes,
                        type_=get_deployment_type(
                            model_config, deployment_id, upstream_endpoint
                        ),
                        upstream_endpoint=upstream_endpoint,
                        upstream_api_key=upstream_config.key,
                    )
                )
        return configs

    def display_config(self) -> str:
        id_ = sanitize_id_part(self.id_)
        if self.upstream_idx is None:
            return id_
        else:
            return f"{id_}/upstream:{self.upstream_idx}"

    @property
    def supports_vision(self):
        types = self.model_attachments or []
        return any(
            ty.startswith("image/") or ty.startswith("*/") for ty in types
        )

    @property
    def supports_video_generation(self):
        if (
            isinstance(self.type_, ChatCompletionDeploymentType)
            and self.type_ == ChatCompletionDeploymentType.AZURE_VIDEO_API
        ):
            return True
        return False

    @property
    def supports_tts(self):
        return self.type_ == ChatCompletionDeploymentType.AUDIO_SPEECH_API

    @property
    def supports_stt(self):
        return (
            self.type_ == ChatCompletionDeploymentType.AUDIO_TRANSCRIPTIONS_API
        )

    @property
    def supports_reasoning(self):
        return self.model_features.reasoningSupported


class TestDeployments(BaseModel):
    __test__ = False

    deployments: List[DeploymentConfig[DeploymentType]]
    app_config: ApplicationConfig

    @classmethod
    def from_config(cls, config_path: str):
        app_config = ApplicationConfig.from_env()
        core_config = CoreConfig.from_config(config_path)

        return cls(
            app_config=app_config,
            deployments=DeploymentConfig.create_deployments(
                core_config,
                functools.partial(_classify_deployment, app_config),
            ),
        )

    @property
    def chat_deployments(
        self,
    ) -> Generator[DeploymentConfig[ChatCompletionDeploymentType], None, None]:
        for deployment in self.deployments:
            if isinstance(deployment.type_, ChatCompletionDeploymentType):
                yield deployment  # type: ignore

    @property
    def embedding_deployments(
        self,
    ) -> Generator[DeploymentConfig[EmbeddingsDeploymentType], None, None]:
        for deployment in self.deployments:
            if isinstance(deployment.type_, EmbeddingsDeploymentType):
                yield deployment  # type: ignore


def sanitize_id_part(value: Any) -> str:
    """Convert any value to a pytest-safe identifier part."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return str(value).replace(".", "p")  # e.g., 0.5 -> 0p5
    if value is None:
        return "none"

    value_str = str(value)
    sanitized = "".join(c if c.isalnum() else "_" for c in value_str)
    return sanitized.strip("_")
