import json
import os
from typing import Dict

from aidial_adapter_openai.utils.parsers import parse_deployment_list
from aidial_adapter_openai.utils.string_processor import (
    StringProcessor,
)


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
DEPLOYMENT_TO_PROMPT_PROCESSOR: Dict[str, StringProcessor] = json.loads(
    os.getenv("DEPLOYMENT_TO_PROMPT_PROCESSOR", "{}")
)
DEPLOYMENT_TO_OUTPUT_PROCESSOR: Dict[str, StringProcessor] = json.loads(
    os.getenv("DEPLOYMENT_TO_OUTPUT_PROCESSOR", "{}")
)
