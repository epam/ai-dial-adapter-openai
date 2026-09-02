from typing import Literal

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
    AnthropicEndpoint,
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


_BedrockClient = Literal["mantle", "runtime"]

_BEDROCK_HOSTS: dict[_BedrockClient, str] = {
    "mantle": "bedrock-mantle.{region}.api.aws",
    "runtime": "bedrock-runtime.{region}.amazonaws.com",
}

_BEDROCK_CLIENTS = list(_BEDROCK_HOSTS)


def _bedrock_base_url(client: _BedrockClient, region: str) -> str:
    return f"https://{_BEDROCK_HOSTS[client].format(region=region)}/openai/v1"


@pytest.mark.parametrize("client", _BEDROCK_CLIENTS)
def test_app_config_chat_responses_bedrock(
    deployment: str, client: _BedrockClient
):
    base_url = _bedrock_base_url(client, "us-east-2")

    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{base_url}/responses"
    )

    assert ty.deployment_type == D.RESPONSES_API
    assert ty.endpoint == BedrockOpenAIEndpoint(
        bedrock_region="us-east-2",
        client=client,
        openai_base_url=base_url,
    )


@pytest.mark.parametrize("client", _BEDROCK_CLIENTS)
def test_app_config_chat_bedrock_chat_completions(
    deployment: str, client: _BedrockClient
):
    base_url = _bedrock_base_url(client, "eu-west-1")

    ty = ApplicationConfig().get_chat_completion_deployment_type(
        deployment, f"{base_url}/chat/completions"
    )

    assert ty.deployment_type == D.GPT_GENERIC
    assert ty.endpoint == BedrockOpenAIEndpoint(
        bedrock_region="eu-west-1",
        client=client,
        openai_base_url=base_url,
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


@pytest.mark.parametrize(
    ("upstream_endpoint", "base_url", "foundry"),
    [
        (
            "https://test.services.ai.azure.com/anthropic/v1/messages",
            "https://test.services.ai.azure.com/anthropic",
            True,
        ),
        (
            "https://api.fireworks.ai/inference/v1/messages",
            "https://api.fireworks.ai/inference",
            False,
        ),
        (
            "https://api.anthropic.com/v1/messages",
            "https://api.anthropic.com",
            False,
        ),
        (
            "https://openrouter.ai/api/v1/messages",
            "https://openrouter.ai/api",
            False,
        ),
        # The /anthropic path alone doesn't make it an Azure Foundry endpoint.
        (
            "https://gateway.example.com/anthropic/v1/messages",
            "https://gateway.example.com/anthropic",
            False,
        ),
    ],
)
def test_app_config_anthropic_messages(
    deployment: str,
    upstream_endpoint: str,
    base_url: str,
    foundry: bool,
):
    cfg = ApplicationConfig()
    ty = cfg.get_chat_completion_deployment_type(deployment, upstream_endpoint)

    assert ty.deployment_type == D.ANTHROPIC_MESSAGES_API
    endpoint = ty.endpoint
    assert isinstance(endpoint, AnthropicEndpoint)
    assert endpoint.anthropic_base_url == base_url
    assert endpoint.foundry == foundry


@pytest.mark.parametrize(
    ("deployment_id", "expected_type"),
    [
        ("vllm.llama3", D.VLLM_CHAT_COMPLETIONS_API),
        ("gpt-4o-mini", D.GPT_GENERIC),
        # the matching is case-sensitive
        ("VLLM.llama3", D.GPT_GENERIC),
    ],
)
def test_deployment_glob_patterns(
    origin: str, deployment_id: str, expected_type: D
):
    cfg = ApplicationConfig(VLLM_DEPLOYMENTS=["vllm.*"])

    ty = cfg.get_chat_completion_deployment_type(
        deployment_id, f"{origin}/whatever/chat/completions"
    )

    assert ty.deployment_type == expected_type


def test_get_vendor_for_generic_deployment(deployment: str):
    cfg = ApplicationConfig()
    assert (
        cfg.get_vendor(
            deployment,
            OpenAIEndpoint(openai_base_url="https://api.openai.com/v1"),
        )
        == Vendor.OPENAI_PLATFORM
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


@pytest.mark.parametrize("client", _BEDROCK_CLIENTS)
def test_get_vendor_for_bedrock_endpoint(
    deployment: str, client: _BedrockClient
):
    region = "us-east-2"
    cfg = ApplicationConfig()
    assert (
        cfg.get_vendor(
            deployment,
            BedrockOpenAIEndpoint(
                bedrock_region=region,
                client=client,
                openai_base_url=_bedrock_base_url(client, region),
            ),
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
