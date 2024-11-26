from typing import Dict, List

from pydantic import BaseModel, Field

import aidial_adapter_openai.env as env


class ApplicationConfig(BaseModel):
    MODEL_ALIASES: Dict[str, str] = Field(default_factory=dict)
    DALLE3_DEPLOYMENTS: List[str] = Field(default_factory=list)
    GPT4_VISION_DEPLOYMENTS: List[str] = Field(default_factory=list)
    MISTRAL_DEPLOYMENTS: List[str] = Field(default_factory=list)
    DATABRICKS_DEPLOYMENTS: List[str] = Field(default_factory=list)
    GPT4O_DEPLOYMENTS: List[str] = Field(default_factory=list)
    GPT4O_MINI_DEPLOYMENTS: List[str] = Field(default_factory=list)
    AZURE_AI_VISION_DEPLOYMENTS: List[str] = Field(default_factory=list)
    API_VERSIONS_MAPPING: Dict[str, str] = Field(default_factory=dict)
    COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES: Dict[str, str] = Field(
        default_factory=dict
    )
    DALLE3_AZURE_API_VERSION: str = Field(default="2024-02-01")
    NON_STREAMING_DEPLOYMENTS: List[str] = Field(default_factory=list)

    @classmethod
    def from_env(cls) -> "ApplicationConfig":
        return cls(
            MODEL_ALIASES=env.MODEL_ALIASES,
            DALLE3_DEPLOYMENTS=env.DALLE3_DEPLOYMENTS,
            GPT4_VISION_DEPLOYMENTS=env.GPT4_VISION_DEPLOYMENTS,
            MISTRAL_DEPLOYMENTS=env.MISTRAL_DEPLOYMENTS,
            DATABRICKS_DEPLOYMENTS=env.DATABRICKS_DEPLOYMENTS,
            GPT4O_DEPLOYMENTS=env.GPT4O_DEPLOYMENTS,
            GPT4O_MINI_DEPLOYMENTS=env.GPT4O_MINI_DEPLOYMENTS,
            AZURE_AI_VISION_DEPLOYMENTS=env.AZURE_AI_VISION_DEPLOYMENTS,
            API_VERSIONS_MAPPING=env.API_VERSIONS_MAPPING,
            COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES=env.COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES,
            DALLE3_AZURE_API_VERSION=env.DALLE3_AZURE_API_VERSION,
            NON_STREAMING_DEPLOYMENTS=env.NON_STREAMING_DEPLOYMENTS,
        )
