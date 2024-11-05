import json
import os
from typing import Dict

from aidial_adapter_openai.utils.env import get_env_bool
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import parse_deployment_list

MODEL_ALIASES: Dict[str, str] = json.loads(os.getenv("MODEL_ALIASES", "{}"))
DALLE3_DEPLOYMENTS = parse_deployment_list(os.getenv("DALLE3_DEPLOYMENTS"))
GPT4_VISION_DEPLOYMENTS = parse_deployment_list(
    os.getenv("GPT4_VISION_DEPLOYMENTS")
)
MISTRAL_DEPLOYMENTS = parse_deployment_list(os.getenv("MISTRAL_DEPLOYMENTS"))
DATABRICKS_DEPLOYMENTS = parse_deployment_list(
    os.getenv("DATABRICKS_DEPLOYMENTS")
)
GPT4O_DEPLOYMENTS = parse_deployment_list(os.getenv("GPT4O_DEPLOYMENTS"))
API_VERSIONS_MAPPING: Dict[str, str] = json.loads(
    os.getenv("API_VERSIONS_MAPPING", "{}")
)
COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES: Dict[str, str] = json.loads(
    os.getenv("COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES") or "{}"
)
DALLE3_AZURE_API_VERSION = os.getenv("DALLE3_AZURE_API_VERSION", "2024-02-01")
NON_STREAMING_DEPLOYMENTS = parse_deployment_list(
    os.getenv("NON_STREAMING_DEPLOYMENTS")
)
AZURE_AI_VISION_DEPLOYMENTS = parse_deployment_list(
    os.getenv("AZURE_AI_VISION_DEPLOYMENTS")
)


def get_eliminate_empty_choices() -> bool:
    old_name = "FIX_STREAMING_ISSUES_IN_NEW_API_VERSIONS"
    new_name = "ELIMINATE_EMPTY_CHOICES"

    if old_name in os.environ:
        logger.warning(
            f"{old_name} environment variable is deprecated. Use {new_name} instead."
        )
        return get_env_bool(old_name, False)

    return get_env_bool(new_name, False)
