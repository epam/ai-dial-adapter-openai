import json
import os
from typing import Dict

from aidial_adapter_openai.utils.parsers import parse_deployment_list

MODEL_ALIASES: Dict[str, str] = json.loads(os.getenv("MODEL_ALIASES", "{}"))
API_VERSIONS_MAPPING: Dict[str, str] = json.loads(
    os.getenv("API_VERSIONS_MAPPING", "{}")
)
DALLE3_AZURE_API_VERSION = os.getenv("DALLE3_AZURE_API_VERSION", "2024-02-01")

# Deployments
DALLE3_DEPLOYMENTS = parse_deployment_list(
    os.getenv("DALLE3_DEPLOYMENTS") or ""
)
GPT4_VISION_DEPLOYMENTS = parse_deployment_list(
    os.getenv("GPT4_VISION_DEPLOYMENTS") or ""
)
MISTRAL_DEPLOYMENTS = parse_deployment_list(
    os.getenv("MISTRAL_DEPLOYMENTS") or ""
)
DATABRICKS_DEPLOYMENTS = parse_deployment_list(
    os.getenv("DATABRICKS_DEPLOYMENTS") or ""
)
GPT4O_DEPLOYMENTS = parse_deployment_list(os.getenv("GPT4O_DEPLOYMENTS") or "")
LEGACY_COMPLETIONS_DEPLOYMENTS = parse_deployment_list(
    os.getenv("LEGACY_COMPLETIONS_DEPLOYMENTS") or ""
)
