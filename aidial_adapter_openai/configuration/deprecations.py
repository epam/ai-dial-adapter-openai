import os

from aidial_adapter_openai.utils.log_config import logger

_gpt4v_is_retired = "GPT-4 Vision models has been deprecated both in Azure OpenAI and on OpenAI Platform"

_deprecated_env_vars = {
    "GPT4_VISION_DEPLOYMENTS": _gpt4v_is_retired,
    "GPT4_VISION_MAX_TOKENS": _gpt4v_is_retired,
}


def check_deprecated_env_vars():
    for name, message in _deprecated_env_vars.items():
        if os.getenv(name) is not None:
            logger.warning(
                f"Environment variable {name!r} is deprecated, since {message}. "
                "The variable could be safely removed from the environment."
            )
