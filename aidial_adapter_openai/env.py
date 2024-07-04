import json
import os
from typing import Dict

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
