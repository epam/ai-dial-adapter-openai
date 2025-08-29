from __future__ import annotations

import os
from typing import Dict, List, assert_never

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as ChatCompletionDeploymentType,
)
from aidial_adapter_openai.configuration.deprecations import (
    check_deprecated_env_vars,
)
from aidial_adapter_openai.utils.env import (
    get_env_bool,
    get_env_dict,
    get_env_list,
    get_env_var,
)
from aidial_adapter_openai.utils.json import remove_nones
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
    chat_completions_parser,
    completions_parser,
    image_gen_parser,
    no_endpoint_parser,
    responses_parser,
)
from aidial_adapter_openai.utils.pydantic import ExtraForbidModel


class DeploymentAPIType(ExtraForbidModel):
    deployment_type: ChatCompletionDeploymentType
    endpoint: AzureOpenAIEndpoint | OpenAIEndpoint


class ApplicationConfig(ExtraForbidModel):
    TIKTOKEN_MODEL_MAPPING: Dict[str, str] = {}

    DALLE3_DEPLOYMENTS: List[str] = []
    DALLE3_AZURE_API_VERSION: str = "2024-02-01"

    GPT_IMAGE_1_DEPLOYMENTS: List[str] = []
    GPT_IMAGE_1_AZURE_API_VERSION: str = "2025-04-01-preview"

    MISTRAL_DEPLOYMENTS: List[str] = []
    DATABRICKS_DEPLOYMENTS: List[str] = []
    GPT4O_DEPLOYMENTS: List[str] = []
    GPT4O_MINI_DEPLOYMENTS: List[str] = []
    AZURE_AI_VISION_DEPLOYMENTS: List[str] = []

    API_VERSIONS_MAPPING: Dict[str, str] = {}
    COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES: Dict[str, str] = {}
    NON_STREAMING_DEPLOYMENTS: List[str] = []
    ELIMINATE_EMPTY_CHOICES: bool = False

    def get_chat_completion_deployment_type(
        self, deployment_id: str, upstream_endpoint: str
    ) -> DeploymentAPIType:
        if deployment_id in self.GPT_IMAGE_1_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.GPT_IMAGE_1,
                endpoint=image_gen_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.DALLE3_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.DALLE3,
                endpoint=image_gen_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.MISTRAL_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.MISTRAL,
                endpoint=no_endpoint_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.DATABRICKS_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.DATABRICKS,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.GPT4O_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.GPT4O,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if deployment_id in self.GPT4O_MINI_DEPLOYMENTS:
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.GPT4O_MINI,
                endpoint=chat_completions_parser.parse(upstream_endpoint),
            )

        if endpoint := completions_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.COMPLETIONS_API,
                endpoint=endpoint,
            )

        if endpoint := responses_parser.try_parse(upstream_endpoint):
            return DeploymentAPIType(
                deployment_type=ChatCompletionDeploymentType.RESPONSES_API,
                endpoint=endpoint,
            )

        return DeploymentAPIType(
            deployment_type=ChatCompletionDeploymentType.GPT_GENERIC,
            endpoint=chat_completions_parser.parse(upstream_endpoint),
        )

    def add_deployment(
        self, deployment_id: str, deployment_type: ChatCompletionDeploymentType
    ) -> ApplicationConfig:
        match deployment_type:
            case ChatCompletionDeploymentType.GPT_IMAGE_1:
                self.GPT_IMAGE_1_DEPLOYMENTS.append(deployment_id)
            case ChatCompletionDeploymentType.DALLE3:
                self.DALLE3_DEPLOYMENTS.append(deployment_id)
            case ChatCompletionDeploymentType.MISTRAL:
                self.MISTRAL_DEPLOYMENTS.append(deployment_id)
            case ChatCompletionDeploymentType.DATABRICKS:
                self.DATABRICKS_DEPLOYMENTS.append(deployment_id)
            case ChatCompletionDeploymentType.GPT4O:
                self.GPT4O_DEPLOYMENTS.append(deployment_id)
            case ChatCompletionDeploymentType.GPT4O_MINI:
                self.GPT4O_MINI_DEPLOYMENTS.append(deployment_id)
            case (
                ChatCompletionDeploymentType.GPT_GENERIC
                | ChatCompletionDeploymentType.RESPONSES_API
                | ChatCompletionDeploymentType.COMPLETIONS_API
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
                "AZURE_AI_VISION_DEPLOYMENTS",
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
                }
            ),
        )
