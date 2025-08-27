import pytest
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
)


@pytest.fixture
def deployment() -> str:
    return "adapter-deployment-name"


@pytest.fixture
def deployment_name() -> str:
    return "upstream-deployment-name:0"


@pytest.fixture(
    params=[
        "https://example.com",
        "http://example.com",
        "http://example.com:8080",
    ],
    ids=["https", "http", "http_port"],
)
def origin(request) -> str:
    return request.param


def test_app_config_chat_openai_platform(origin: str, deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment,
        f"{origin}/whatever1/whatever2/chat/completions",
    )

    assert ty.deployment_type == D.GPT_GENERIC
    endpoint = ty.endpoint
    assert isinstance(endpoint, OpenAIEndpoint)
    assert endpoint.base_url == f"{origin}/whatever1/whatever2"


def test_app_config_chat_azure(
    origin: str, deployment: str, deployment_name: str
):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment,
        f"{origin}/whatever1/whatever2/openai/deployments/{deployment_name}/chat/completions",
    )

    assert ty.deployment_type == D.GPT_GENERIC
    assert ty.endpoint == AzureOpenAIEndpoint(
        azure_endpoint=f"{origin}/whatever1/whatever2",
        azure_deployment=deployment_name,
    )


def test_app_config_chat_responses_azure_prev_gen(origin: str, deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{origin}/whatever1/whatever2/openai/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == AzureOpenAIEndpoint(
        azure_endpoint=f"{origin}/whatever1/whatever2", next_gen_api=False
    )


def test_app_config_chat_responses_azure_next_gen(origin: str, deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{origin}/whatever1/whatever2/openai/v1/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == AzureOpenAIEndpoint(
        azure_base_url=f"{origin}/whatever1/whatever2/openai/v1",
        next_gen_api=True,
    )


def test_app_config_chat_responses_openai_platform(
    origin: str, deployment: str
):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{origin}/whatever1/whatever2/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == OpenAIEndpoint(
        base_url=f"{origin}/whatever1/whatever2"
    )


def test_app_config_chat_invalid(origin: str, deployment: str):
    """Responses deployment misconfigured as GPT-4o"""

    with pytest.raises(InvalidRequestError) as exc_info:
        (
            ApplicationConfig()
            .add_deployment(deployment, D.GPT4O)
            .get_chat_completion_deployment_type(
                deployment, f"{origin}/whatever1/whatever2/responses"
            )
        )

    error = exc_info.value
    assert error.message == "Invalid upstream endpoint format"


def test_app_config_dalle_azure(
    origin: str, deployment: str, deployment_name: str
):
    ty = (
        ApplicationConfig()
        .add_deployment(deployment, D.DALLE3)
        .get_chat_completion_deployment_type(
            deployment,
            f"{origin}/whatever1/whatever2/openai/deployments/{deployment_name}/images/generations",
        )
    )

    assert ty.deployment_type == D.DALLE3
    assert ty.endpoint == AzureOpenAIEndpoint(
        azure_endpoint=f"{origin}/whatever1/whatever2",
        azure_deployment=deployment_name,
    )
