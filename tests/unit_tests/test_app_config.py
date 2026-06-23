import pytest
from aidial_sdk.exceptions import HTTPException

from aidial_adapter_openai.configuration.app_config import (
    ApplicationConfig,
    Vendor,
)
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    BedrockOpenAIEndpoint,
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
    assert endpoint.openai_base_url == f"{origin}/whatever1/whatever2"


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
        azure_endpoint=f"{origin}/whatever1/whatever2"
    )


def test_app_config_chat_responses_azure_next_gen(origin: str, deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{origin}/whatever1/whatever2/openai/v1/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == OpenAIEndpoint(
        openai_base_url=f"{origin}/whatever1/whatever2/openai/v1"
    )


def test_app_config_chat_responses_openai_platform(
    origin: str, deployment: str
):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{origin}/whatever1/whatever2/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == OpenAIEndpoint(
        openai_base_url=f"{origin}/whatever1/whatever2"
    )


def test_app_config_chat_responses_bedrock(deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment,
        "https://bedrock-mantle.us-east-2.api.aws/openai/v1/responses",
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == BedrockOpenAIEndpoint(
        bedrock_region="us-east-2",
    )


def test_app_config_chat_bedrock_chat_completions(deployment: str):
    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment,
        "https://bedrock-mantle.eu-west-1.api.aws/openai/v1/chat/completions",
    )

    assert ty.deployment_type == D.GPT_GENERIC
    assert ty.endpoint == BedrockOpenAIEndpoint(
        bedrock_region="eu-west-1",
    )


def test_app_config_chat_completions_azure_next_gen(
    origin: str, deployment: str
):
    ty = (
        ApplicationConfig()
        .add_deployment(deployment, D.GPT4O)
        .get_chat_completion_deployment_type(
            deployment,
            f"{origin}/whatever1/whatever2/openai/v1/chat/completions",
        )
    )

    assert ty.deployment_type == D.GPT4O
    assert ty.endpoint == OpenAIEndpoint(
        openai_base_url=f"{origin}/whatever1/whatever2/openai/v1"
    )


def test_app_config_chat_invalid(origin: str, deployment: str):
    with pytest.raises(HTTPException) as exc_info:
        (
            ApplicationConfig()
            .add_deployment(deployment, D.GPT4O)
            .get_chat_completion_deployment_type(
                deployment, f"{origin}/whatever1/whatever2/whatever3"
            )
        )

    error = exc_info.value
    assert error.status_code == 502
    assert error.code == "502"
    assert error.type == "internal_server_error"
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


def test_app_config_qwen3_asr_vllm_deployments(origin: str, deployment: str):
    cfg = ApplicationConfig(QWEN3_ASR_VLLM_DEPLOYMENTS=[deployment])

    ty = cfg.get_chat_completion_deployment_type(
        deployment,
        f"{origin}/whatever1/whatever2/chat/completions",
    )

    assert ty.deployment_type == D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API


def test_get_vendor_for_generic_deployment(deployment: str):
    cfg = ApplicationConfig()
    assert (
        cfg.get_vendor(
            deployment,
            OpenAIEndpoint(openai_base_url="https://api.openai.com/v1"),
        )
        == Vendor.VLLM
    )


@pytest.mark.parametrize(
    "cfg",
    [
        ApplicationConfig(VLLM_DEPLOYMENTS=["vllm-deployment"]),
        ApplicationConfig(
            QWEN3_ASR_VLLM_DEPLOYMENTS=["qwen3-asr-vllm-deployment"]
        ),
    ],
)
def test_get_vendor_for_vllm_families(cfg: ApplicationConfig):
    deployment_id = next(
        iter(cfg.VLLM_DEPLOYMENTS or cfg.QWEN3_ASR_VLLM_DEPLOYMENTS)
    )
    assert (
        cfg.get_vendor(
            deployment_id,
            OpenAIEndpoint(openai_base_url="https://api.openai.com/v1"),
        )
        == Vendor.VLLM
    )


def test_get_vendor_for_bedrock_endpoint(deployment: str):
    cfg = ApplicationConfig()
    assert (
        cfg.get_vendor(
            deployment,
            BedrockOpenAIEndpoint(bedrock_region="us-east-2"),
        )
        == Vendor.AWS
    )


def test_is_azure_for_generic_deployment(deployment: str):
    cfg = ApplicationConfig()
    assert cfg.is_azure(deployment)


@pytest.mark.parametrize(
    "cfg",
    [
        ApplicationConfig(VLLM_DEPLOYMENTS=["vllm-deployment"]),
        ApplicationConfig(
            QWEN3_ASR_VLLM_DEPLOYMENTS=["qwen3-asr-vllm-deployment"]
        ),
    ],
)
def test_is_azure_for_non_azure_vllm_families(cfg: ApplicationConfig):
    deployment_id = next(
        iter(cfg.VLLM_DEPLOYMENTS or cfg.QWEN3_ASR_VLLM_DEPLOYMENTS)
    )
    assert not cfg.is_azure(deployment_id)
