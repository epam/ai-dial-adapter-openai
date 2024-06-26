import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
    chat_completions_parser,
    completions_parser,
)

NORMAL_CHAT_CASES = [
    (
        "https://test.com/openai/deployments/test-deployment/chat/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/my/models/openai/deployments/test-deployment/chat/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com/my/models",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/openai/deployments/test-deployment-chat/chat/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment-chat",
        ),
    ),
    (
        "https://test.com/openai/deployments/chat/completions",
        OpenAIEndpoint(
            base_url="https://test.com/openai/deployments",
        ),
    ),
    (
        "https://test.com/my/endpoint/chat/completions",
        OpenAIEndpoint(
            base_url="https://test.com/my/endpoint",
        ),
    ),
]


FAILING_CHAT_CASES = [
    "https://test.com/openai/deployments/chat/completion",
    "https://test.com/openai/api/completions",
    "https://test.com/openai/deployments/test-deployment/completions",
]

NORMAL_COMPLETIONS_CASES = [
    (
        "https://test.com/openai/deployments/test-deployment/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/my/models/openai/deployments/test-deployment/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com/my/models",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/openai/deployments/test-deployment-chat/completions",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment-chat",
        ),
    ),
    (
        "https://test.com/openai/deployments/completions",
        OpenAIEndpoint(
            base_url="https://test.com/openai/deployments",
        ),
    ),
    (
        "https://test.com/my/endpoint/completions",
        OpenAIEndpoint(
            base_url="https://test.com/my/endpoint",
        ),
    ),
]


@pytest.mark.parametrize("endpoint, parsed", NORMAL_CHAT_CASES)
def test_chat_completions_normal(endpoint, parsed):
    result = chat_completions_parser.parse(endpoint)
    assert result == parsed


@pytest.mark.parametrize("endpoint", FAILING_CHAT_CASES)
def test_chat_failing_cases(endpoint):
    with pytest.raises(DialException):
        chat_completions_parser.parse(endpoint)


@pytest.mark.parametrize("endpoint, parsed", NORMAL_COMPLETIONS_CASES)
def test_completions_parser_normal(endpoint, parsed):
    result = completions_parser.parse(endpoint)
    assert result == parsed


@pytest.mark.parametrize("endpoint, parsed", NORMAL_CHAT_CASES)
def test_completions_parser_invalid(endpoint, parsed):
    result = completions_parser.parse(endpoint)
    assert result is None
