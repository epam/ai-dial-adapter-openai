import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.utils import auth


@pytest.mark.asyncio
async def test_get_credentials_returns_api_key_when_present_for_azure_true():
    creds = await auth.get_credentials(
        {"X-UPSTREAM-KEY": "test-key"}, azure=True
    )

    assert creds == {"api_key": "test-key"}


@pytest.mark.asyncio
async def test_get_credentials_returns_api_key_when_present_for_azure_false():
    creds = await auth.get_credentials(
        {"X-UPSTREAM-KEY": "test-key"}, azure=False
    )

    assert creds == {"api_key": "test-key"}


@pytest.mark.asyncio
async def test_get_credentials_falls_back_to_azure_ad_token(monkeypatch):
    async def _mock_get_api_key() -> str:
        return "token-123"

    monkeypatch.setattr(auth, "get_azure_access_token", _mock_get_api_key)

    creds = await auth.get_credentials({}, azure=True)

    assert creds == {"azure_ad_token": "token-123"}


@pytest.mark.asyncio
async def test_get_credentials_raises_without_key_for_non_azure():
    with pytest.raises(DialException) as exc_info:
        await auth.get_credentials({}, azure=False)

    error = exc_info.value
    assert error.status_code == 401
    assert error.message == "X-UPSTREAM-KEY header is missing"
