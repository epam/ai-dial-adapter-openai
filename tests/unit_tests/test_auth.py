import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.configuration.app_config import Vendor
from aidial_adapter_openai.utils import auth
from aidial_adapter_openai.utils.parsers import BedrockOpenAIEndpoint


@pytest.mark.asyncio
async def test_get_credentials_returns_api_key_when_present_for_azure_true():
    creds = await auth.get_credentials(
        {"X-UPSTREAM-KEY": "test-key"}, vendor=Vendor.AZURE, endpoint=None
    )

    assert creds == {"api_key": "test-key"}


@pytest.mark.asyncio
async def test_get_credentials_returns_api_key_when_present_for_azure_false():
    creds = await auth.get_credentials(
        {"X-UPSTREAM-KEY": "test-key"}, vendor=Vendor.VLLM, endpoint=None
    )

    assert creds == {"api_key": "test-key"}


@pytest.mark.asyncio
async def test_get_credentials_falls_back_to_azure_ad_token(monkeypatch):
    class _mock_token_provider:
        async def aclose(self):
            pass

        async def get_token(self) -> str:
            return "token-123"

    monkeypatch.setattr(auth, "_AzureTokenProvider", _mock_token_provider)

    creds = await auth.get_credentials({}, vendor=Vendor.AZURE, endpoint=None)

    assert creds == {"azure_ad_token": "token-123"}


@pytest.mark.asyncio
async def test_get_credentials_raises_without_key_for_non_azure():
    with pytest.raises(DialException) as exc_info:
        await auth.get_credentials({}, vendor=Vendor.VLLM, endpoint=None)

    error = exc_info.value
    assert error.status_code == 401
    assert error.message == "X-UPSTREAM-KEY header is missing"


@pytest.mark.asyncio
async def test_get_credentials_allows_empty_for_aws_vendor(monkeypatch):
    for env_var in (
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_ASSUME_ROLE_ARN",
    ):
        monkeypatch.delenv(env_var, raising=False)

    endpoint = BedrockOpenAIEndpoint(bedrock_region="us-east-1")
    creds = await auth.get_credentials({}, vendor=Vendor.AWS, endpoint=endpoint)
    assert creds == {
        "aws_access_key_id": None,
        "aws_secret_access_key": None,
        "aws_session_token": None,
    }
