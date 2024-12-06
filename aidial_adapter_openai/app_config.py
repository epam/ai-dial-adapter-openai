import json
import os
from typing import Dict, List

from pydantic import BaseModel

from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from aidial_adapter_openai.utils.env import get_env_bool
from aidial_adapter_openai.utils.json import remove_nones
from aidial_adapter_openai.utils.log_config import logger


class ApplicationConfig(BaseModel):
    MODEL_ALIASES: Dict[str, str] = {}
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

    def get_chat_completion_deployment_type(
        self, deployment_id: str
    ) -> ChatCompletionDeploymentType:
        if deployment_id in self.DALLE3_DEPLOYMENTS:
            return ChatCompletionDeploymentType.DALLE3
        elif deployment_id in self.GPT4_VISION_DEPLOYMENTS:
            return ChatCompletionDeploymentType.GPT4_VISION
        elif deployment_id in self.MISTRAL_DEPLOYMENTS:
            return ChatCompletionDeploymentType.MISTRAL
        elif deployment_id in self.DATABRICKS_DEPLOYMENTS:
            return ChatCompletionDeploymentType.DATABRICKS
        elif deployment_id in self.GPT4O_DEPLOYMENTS:
            return ChatCompletionDeploymentType.GPT4O
        elif deployment_id in self.GPT4O_MINI_DEPLOYMENTS:
            return ChatCompletionDeploymentType.GPT4O_MINI
        else:
            return ChatCompletionDeploymentType.GPT_TEXT_ONLY

    @classmethod
    def from_env(cls) -> "ApplicationConfig":
        def _parse_env_deployments(deployments_key: str) -> List[str] | None:
            deployments_value = os.getenv(deployments_key)
            if deployments_value is None:
                return None
            return list(map(str.strip, (deployments_value).split(",")))

        def _parse_env_dict(key: str) -> Dict[str, str] | None:
            value = os.getenv(key)
            return json.loads(value) if value else None

        def _parse_eliminate_empty_choices() -> bool | None:
            old_name = "FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS"
            new_name = "ELIMINATE_EMPTY_CHOICES"

            if old_name in os.environ:
                logger.warning(
                    f"{old_name} environment variable is deprecated. Use {new_name} instead."
                )
                return get_env_bool(old_name)
            elif new_name in os.environ:
                return get_env_bool(new_name)

            return None

        deployment_fields = {
            deployment_key: _parse_env_deployments(deployment_key)
            for deployment_key in (
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
            key: _parse_env_dict(key)
            for key in (
                "MODEL_ALIASES",
                "API_VERSIONS_MAPPING",
                "COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES",
            )
        }

        return cls(
            **remove_nones(
                {
                    **deployment_fields,
                    **dict_fields,
                    "DALLE3_AZURE_API_VERSION": os.getenv(
                        "DALLE3_AZURE_API_VERSION"
                    ),
                    "ELIMINATE_EMPTY_CHOICES": _parse_eliminate_empty_choices(),
                }
            ),
        )
