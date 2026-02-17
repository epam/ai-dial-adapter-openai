import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    OpenAIEndpoint,
    chat_completions_parser,
    completions_parser,
    no_endpoint_parser,
    openai_video_api_parser,
    responses_parser,
)
from tests.conftest import create_test_client

RESPONSE_CASES = [
    (
        "https://test.com/openai/v1/responses",
        OpenAIEndpoint(base_url="https://test.com/openai/v1"),
    ),
    (
        "https://test.com/openai/responses",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com", azure_deployment=None
        ),
    ),
]

NO_ENDPOINT_CASES = [
    (
        "https://test.com/openai/deployments/test-deployment",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/my/models/openai/deployments/test-deployment",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com/my/models",
            azure_deployment="test-deployment",
        ),
    ),
    (
        "https://test.com/openai/deployments/test-deployment-chat",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment-chat",
        ),
    ),
    (
        "https://test.com/openai/deployments",
        OpenAIEndpoint(base_url="https://test.com/openai/deployments"),
    ),
    (
        "https://test.com/my/endpoint",
        OpenAIEndpoint(base_url="https://test.com/my/endpoint"),
    ),
]


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
    result = completions_parser.try_parse(endpoint)
    assert result == parsed


@pytest.mark.parametrize("endpoint, parsed", NORMAL_CHAT_CASES)
def test_completions_parser_invalid(endpoint, parsed):
    result = completions_parser.try_parse(endpoint)
    assert result is None


@pytest.mark.parametrize("endpoint, parsed", RESPONSE_CASES)
def test_responses_parser(endpoint, parsed):
    result = responses_parser.try_parse(endpoint)
    assert result == parsed


@pytest.mark.parametrize("endpoint, parsed", NO_ENDPOINT_CASES)
def test_no_endpoint_parser(endpoint, parsed):
    result = no_endpoint_parser.try_parse(endpoint)
    assert result == parsed


OPENAI_VIDEO_API_CASES = [
    (
        "https://test.com/openai/v1/videos",
        OpenAIEndpoint(base_url="https://test.com/openai/v1"),
    ),
    (
        "https://test.com/openai/deployments/test-deployment/videos",
        AzureOpenAIEndpoint(
            azure_endpoint="https://test.com",
            azure_deployment="test-deployment",
        ),
    ),
]


@pytest.mark.parametrize("endpoint, parsed", OPENAI_VIDEO_API_CASES)
def test_openai_video_api_parser(endpoint, parsed):
    result = openai_video_api_parser.try_parse(endpoint)
    assert result == parsed


@pytest.mark.parametrize("stream", [True, False], ids=["stream", "block"])
async def test_openai_video_api_only_v1_api_supported(stream: bool):
    async with create_test_client(ApplicationConfig()) as client:
        response = await client.post(
            "/openai/deployments/app/chat/completions?api-version=2023-03-15-preview",
            json={"messages": [], "stream": stream},
            headers={
                "X-UPSTREAM-KEY": "TEST_API_KEY",
                "X-UPSTREAM-ENDPOINT": "http://localhost:5001/openai/deployments/upstream-model/videos",
            },
        )

        assert response.status_code == 502
        assert response.json() == {
            "error": {
                "type": "internal_server_error",
                "code": "502",
                "message": "Invalid upstream endpoint format: Only v1 API upstream endpoints are supported for OpenAI video generation deployments",
            }
        }
