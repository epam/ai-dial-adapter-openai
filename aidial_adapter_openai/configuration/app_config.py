from __future__ import annotations

import os
from enum import StrEnum
from fnmatch import fnmatchcase
from typing import Annotated, assert_never

from aidial_sdk.exceptions import InternalServerError
from pydantic import AfterValidator

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.configuration.deprecations import (
    check_deprecated_env_vars,
)
from aidial_adapter_openai.utils.env import (
    get_env_bool,
    get_env_dict,
    get_env_float,
    get_env_list,
    get_env_var,
)
from aidial_adapter_openai.utils.json import remove_nones
from aidial_adapter_openai.utils.parsers import (
    AnthropicEndpoint,
    AzureOpenAIEndpoint,
    BedrockOpenAIEndpoint,
    OpenAIEndpoint,
    anthropic_messages_parser,
    azure_video_api_parser,
    chat_completions_optional_parser,
    chat_completions_parser,
    completions_parser,
    image_gen_parser,
    openai_video_api_parser,
    responses_parser,
    speech_parser,
    transcriptions_parser,
)
from aidial_adapter_openai.utils.pydantic import ExtraForbidModel


class DeploymentPatterns(list[str]):
    def __contains__(self, deployment_id: object) -> bool:
        return isinstance(deployment_id, str) and any(
            fnmatchcase(deployment_id, pattern) for pattern in self
        )


DeploymentList = Annotated[list[str], AfterValidator(DeploymentPatterns)]


class DeploymentAPIType(ExtraForbidModel):
    deployment_type: D
    endpoint: (
        AzureOpenAIEndpoint
        | OpenAIEndpoint
        | AnthropicEndpoint
        | BedrockOpenAIEndpoint
    )


class Vendor(StrEnum):
    AWS = "aws"
    VLLM = "vllm"
    AZURE = "azure"
    OPENAI_PLATFORM = "openai_platform"
    ALIBABA = "alibaba"


class ApplicationConfig(ExtraForbidModel):
    TIKTOKEN_MODEL_MAPPING: dict[str, str] = {}

    DALLE3_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    DALLE3_AZURE_API_VERSION: str = "2024-02-01"

    GPT_IMAGE_1_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    GPT_IMAGE_1_AZURE_API_VERSION: str = "2025-04-01-preview"

    MISTRAL_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    DATABRICKS_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    GPT4O_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    GPT4O_MINI_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    VLLM_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    QWEN3_ASR_VLLM_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    AZURE_AI_VISION_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    ALIBABA_DEPLOYMENTS: DeploymentList = DeploymentPatterns()

    API_VERSIONS_MAPPING: dict[str, str] = {}
    COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES: dict[str, str] = {}
    NON_STREAMING_DEPLOYMENTS: DeploymentList = DeploymentPatterns()
    ELIMINATE_EMPTY_CHOICES: bool = False
    SSE_HEARTBEAT_INTERVAL: float | None = None

    AUDIO_AZURE_API_VERSION: str = "2025-03-01-preview"

    def is_azure(self, deployment_id: str) -> bool:
        for deployments in [
            self.VLLM_DEPLOYMENTS,
            self.QWEN3_ASR_VLLM_DEPLOYMENTS,
        ]:
            if deployment_id in deployments:
                return False
        return True

    def get_vendor(
        self,
        deployment_id: str | None,
        endpoint: (
            AzureOpenAIEndpoint
            | OpenAIEndpoint
            | AnthropicEndpoint
            | BedrockOpenAIEndpoint
            | None
        ),
    ) -> Vendor:
        if isinstance(endpoint, BedrockOpenAIEndpoint):
            return Vendor.AWS

        for deployments in [
            self.VLLM_DEPLOYMENTS,
            self.QWEN3_ASR_VLLM_DEPLOYMENTS,
        ]:
            if deployment_id in deployments:
                return Vendor.VLLM

        if deployment_id in self.ALIBABA_DEPLOYMENTS:
            return Vendor.ALIBABA

        if (
            isinstance(endpoint, OpenAIEndpoint)
            and "api.openai.com" in endpoint.openai_base_url
        ):
            return Vendor.OPENAI_PLATFORM

        return Vendor.AZURE

    def get_chat_completion_deployment_type(
        self, deployment_id: str, upstream_endpoint: str
    ) -> DeploymentAPIType:
        if endpoint := anthropic_messages_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.ANTHROPIC_MESSAGES_API,
                endpoint=endpoint,
            )

        if endpoint := completions_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.COMPLETIONS_API,
                endpoint=endpoint,
            )

        if endpoint := responses_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.RESPONSES_API,
                endpoint=endpoint,
            )

        if endpoint := openai_video_api_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.OPENAI_VIDEO_API,
                endpoint=endpoint,
            )

        if endpoint := azure_video_api_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.AZURE_VIDEO_API,
                endpoint=endpoint,
            )

        if endpoint := speech_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.AUDIO_SPEECH_API,
                endpoint=endpoint,
            )

        if endpoint := transcriptions_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=D.AUDIO_TRANSCRIPTIONS_API,
                endpoint=endpoint,
            )

        if endpoint := image_gen_parser.try_parse(upstream_endpoint):
            if deployment_id in self.GPT_IMAGE_1_DEPLOYMENTS:
                return DeploymentAPIType(
                    deployment_type=D.GPT_IMAGE_1,
                    endpoint=endpoint,
                )
            elif deployment_id in self.DALLE3_DEPLOYMENTS:
                return DeploymentAPIType(
                    deployment_type=D.DALLE3,
                    endpoint=endpoint,
                )
            else:
                raise InternalServerError(
                    f"The image generation deployment id {deployment_id!r} must be "
                    "declared either in GPT_IMAGE_1_DEPLOYMENTS or DALLE3_DEPLOYMENTS env variable."
                )
        elif (
            deployment_id in self.GPT_IMAGE_1_DEPLOYMENTS
            or deployment_id in self.DALLE3_DEPLOYMENTS
        ):
            raise InternalServerError(
                f"Image generation deployment id ({deployment_id!r}) declared "
                "in GPT_IMAGE_1_DEPLOYMENTS or DALLE3_DEPLOYMENTS "
                "env variable must use images/generations upstream endpoint."
            )

        if deployment_id in self.MISTRAL_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.MISTRAL,
                endpoint=chat_completions_optional_parser.parse(
                    upstream_endpoint
                ),
            )

        if deployment_id in self.DATABRICKS_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.DATABRICKS,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.VLLM_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.VLLM_CHAT_COMPLETIONS_API,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.QWEN3_ASR_VLLM_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.GPT4O_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.GPT4O,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.GPT4O_MINI_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=D.GPT4O_MINI,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        return DeploymentAPIType(
            deployment_type=D.GPT_GENERIC,
            endpoint=chat_completions_parser.parse(upstream_endpoint),
        )

    def add_deployment(
        self, deployment_id: str, deployment_type: D
    ) -> ApplicationConfig:
        match deployment_type:
            case D.GPT_IMAGE_1:
                self.GPT_IMAGE_1_DEPLOYMENTS.append(deployment_id)
            case D.DALLE3:
                self.DALLE3_DEPLOYMENTS.append(deployment_id)
            case D.MISTRAL:
                self.MISTRAL_DEPLOYMENTS.append(deployment_id)
            case D.DATABRICKS:
                self.DATABRICKS_DEPLOYMENTS.append(deployment_id)
            case D.GPT4O:
                self.GPT4O_DEPLOYMENTS.append(deployment_id)
            case D.GPT4O_MINI:
                self.GPT4O_MINI_DEPLOYMENTS.append(deployment_id)
            case D.VLLM_CHAT_COMPLETIONS_API:
                self.VLLM_DEPLOYMENTS.append(deployment_id)
            case D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API:
                self.QWEN3_ASR_VLLM_DEPLOYMENTS.append(deployment_id)
            case (
                D.GPT_GENERIC
                | D.RESPONSES_API
                | D.COMPLETIONS_API
                | D.AZURE_VIDEO_API
                | D.AUDIO_SPEECH_API
                | D.AUDIO_TRANSCRIPTIONS_API
                | D.OPENAI_VIDEO_API
                | D.ANTHROPIC_MESSAGES_API
            ):
                pass
            case _:
                assert_never(deployment_type)
        return self

    def map_to_tiktoken_model(
        self, deployment_id: str, tiktoken_model: str
    ) -> ApplicationConfig:
        self.TIKTOKEN_MODEL_MAPPING[deployment_id] = tiktoken_model
        return self

    @classmethod
    def from_env(cls) -> ApplicationConfig:
        check_deprecated_env_vars()

        list_fields = {
            key: get_env_var(get_env_list, key)
            for key in (
                "DALLE3_DEPLOYMENTS",
                "GPT_IMAGE_1_DEPLOYMENTS",
                "MISTRAL_DEPLOYMENTS",
                "DATABRICKS_DEPLOYMENTS",
                "GPT4O_DEPLOYMENTS",
                "GPT4O_MINI_DEPLOYMENTS",
                "VLLM_DEPLOYMENTS",
                "QWEN3_ASR_VLLM_DEPLOYMENTS",
                "AZURE_AI_VISION_DEPLOYMENTS",
                "ALIBABA_DEPLOYMENTS",
                "NON_STREAMING_DEPLOYMENTS",
            )
        }

        dict_fields = {
            key: get_env_var(get_env_dict, key)
            for key in (
                "API_VERSIONS_MAPPING",
                "COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES",
            )
        }

        return cls(
            **remove_nones(
                {
                    **list_fields,
                    **dict_fields,
                    "DALLE3_AZURE_API_VERSION": get_env_var(
                        os.getenv, "DALLE3_AZURE_API_VERSION"
                    ),
                    "GPT_IMAGE_1_AZURE_API_VERSION": get_env_var(
                        os.getenv, "GPT_IMAGE_1_AZURE_API_VERSION"
                    ),
                    "AUDIO_AZURE_API_VERSION": get_env_var(
                        os.getenv, "AUDIO_AZURE_API_VERSION"
                    ),
                    "ELIMINATE_EMPTY_CHOICES": get_env_var(
                        get_env_bool,
                        "ELIMINATE_EMPTY_CHOICES",
                        deprecated_names=[
                            "FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS"
                        ],
                    ),
                    "TIKTOKEN_MODEL_MAPPING": get_env_var(
                        get_env_dict,
                        "TIKTOKEN_MODEL_MAPPING",
                        deprecated_names=["MODEL_ALIASES"],
                    ),
                    "SSE_HEARTBEAT_INTERVAL": get_env_float(
                        "SSE_HEARTBEAT_INTERVAL"
                    ),
                }
            ),
        )
