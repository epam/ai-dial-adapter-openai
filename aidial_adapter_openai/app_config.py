import os
from typing import Callable, Dict, List

from pydantic import BaseModel

from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from aidial_adapter_openai.utils.env import (
    get_env_bool,
    get_env_dict,
    get_env_list,
    get_env_var,
)
from aidial_adapter_openai.utils.json import remove_nones


class ApplicationConfig(BaseModel):
    TIKTOKEN_MODEL_MAPPING: Dict[str, str] = {}
    DALLE3_DEPLOYMENTS: List[str] = []
    GPT4_VISION_DEPLOYMENTS: List[str] = []
    MISTRAL_DEPLOYMENTS: List[str] = []
    DATABRICKS_DEPLOYMENTS: List[str] = []
    GPT4O_DEPLOYMENTS: List[str] = []
    GPT4O_MINI_DEPLOYMENTS: List[str] = []
    AZURE_AI_VISION_DEPLOYMENTS: List[str] = []
    API_VERSIONS_MAPPING: Dict[str, str] = {}
    COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES: Dict[str, str] = {}
    DALLE3_AZURE_API_VERSION: str = "2024-02-01"
    NON_STREAMING_DEPLOYMENTS: List[str] = []
    ELIMINATE_EMPTY_CHOICES: bool = False

    _DEPLOYMENT_TYPE_MAP: Dict[
        ChatCompletionDeploymentType, Callable[["ApplicationConfig"], List[str]]
    ] = {
        ChatCompletionDeploymentType.DALLE3: lambda config: config.DALLE3_DEPLOYMENTS,
        ChatCompletionDeploymentType.GPT4_VISION: lambda config: config.GPT4_VISION_DEPLOYMENTS,
        ChatCompletionDeploymentType.MISTRAL: lambda config: config.MISTRAL_DEPLOYMENTS,
        ChatCompletionDeploymentType.DATABRICKS: lambda config: config.DATABRICKS_DEPLOYMENTS,
        ChatCompletionDeploymentType.GPT4O: lambda config: config.GPT4O_DEPLOYMENTS,
        ChatCompletionDeploymentType.GPT4O_MINI: lambda config: config.GPT4O_MINI_DEPLOYMENTS,
    }

    def get_chat_completion_deployment_type(
        self, deployment_id: str
    ) -> ChatCompletionDeploymentType:
        for deployment_type, config_getter in self._DEPLOYMENT_TYPE_MAP.items():
            if deployment_id in config_getter(self):
                return deployment_type
        return ChatCompletionDeploymentType.GPT_TEXT_ONLY

    def add_deployment(
        self, deployment_id: str, deployment_type: ChatCompletionDeploymentType
    ) -> "ApplicationConfig":
        if deployment_type != ChatCompletionDeploymentType.GPT_TEXT_ONLY:
            config_getter = self._DEPLOYMENT_TYPE_MAP[deployment_type]
            config_getter(self).append(deployment_id)
        return self

    def map_to_tiktoken_model(
        self, deployment_id: str, tiktoken_model: str
    ) -> "ApplicationConfig":
        self.TIKTOKEN_MODEL_MAPPING[deployment_id] = tiktoken_model
        return self

    @classmethod
    def from_env(cls) -> "ApplicationConfig":

        list_fields = {
            key: get_env_var(get_env_list, key)
            for key in (
                "DALLE3_DEPLOYMENTS",
                "GPT4_VISION_DEPLOYMENTS",
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
