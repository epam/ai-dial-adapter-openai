from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.utils.parsers import OpenAIEndpoint


def test_deployment_api_type():
    app_config = ApplicationConfig()
    deployment_type = app_config.get_chat_completion_deployment_type(
        "my-deployment",
        "https://example.com/models/chat/completions",
    )
    assert (
        deployment_type.deployment_type
        == ChatCompletionDeploymentType.GPT_TEXT_ONLY
    )
    endpoint = deployment_type.endpoint
    assert isinstance(endpoint, OpenAIEndpoint)
    assert endpoint.base_url == "https://example.com/models"
