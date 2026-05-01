import os

from aidial_adapter_openai.utils.log_config import logger

_gpt4_deprecation_message = "GPT-4 Vision models has been deprecated both in Azure OpenAI and on OpenAI Platform."

_deprecated_env_vars = {
    "GPT4_VISION_DEPLOYMENTS": (
        f"{_gpt4_deprecation_message} "
        """
In the unlikely case that the deployments declared under this variable are still in use:
1. move them to the GPT4O_DEPLOYMENTS variable,
2. map the deployments to "gpt-4" in the TIKTOKEN_MODEL_MAPPING variable,
3. remove the deprecated variable.""".strip()
    ),
    "GPT4_VISION_MAX_TOKENS": (
        f"{_gpt4_deprecation_message} "
        f"The variable is of no use, it could be safely removed."
    ),
    "DIAL_USE_FILE_STORAGE": (
        "Previously, 'DIAL_USE_FILE_STORAGE=True' explicitly enabled DIAL Storage, and 'DIAL_USE_FILE_STORAGE=False' or leaving it unset disabled it. "
        "After the change, DIAL Storage is enabled automatically when 'DIAL_URL' is set. 'DIAL_USE_FILE_STORAGE' is deprecated and no longer forces storage to be enabled when 'DIAL_URL' is missing. "
        "'DIAL_USE_FILE_STORAGE' is of no use, it could be safely removed."
    ),
}


def check_deprecated_env_vars():
    for name, message in _deprecated_env_vars.items():
        if os.getenv(name) is not None:
            logger.warning(
                f"Environment variable {name!r} is deprecated. {message}"
            )
