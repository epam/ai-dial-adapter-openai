import json
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.configuration.app_config import (
    Vendor,
)
from aidial_adapter_openai.utils import auth
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import BedrockOpenAIEndpoint
from aidial_adapter_openai.utils.upstream_headers import (
    UPSTREAM_EXTRA_DATA_HEADER,
)

_REGION = "us-east-1"
_ENDPOINT = BedrockOpenAIEndpoint(bedrock_region=_REGION)
_ROLE_ARN = "arn:aws:iam::123456789012:role/BedrockAccess"
_AWS_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ASSUME_ROLE_ARN",
)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vendor", list(Vendor), ids=lambda vendor: vendor.value
)
async def test_get_credentials_returns_api_key_for_any_vendor(vendor: Vendor):
    creds = await auth.get_credentials(
        {"X-UPSTREAM-KEY": "test-key"}, vendor=vendor, endpoint=None
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


@pytest.fixture(autouse=True)
def isolate_aws_environment(monkeypatch: pytest.MonkeyPatch):
    """
    The adapter falls back to these variables, so the environment of whoever
    runs the tests must not leak into them.
    """
    for env_var in _AWS_ENV_VARS:
        monkeypatch.delenv(env_var, raising=False)


@pytest.fixture(autouse=True)
async def reset_assume_role_providers():
    await auth.get_assume_role_provider.clear()
    yield
    await auth.get_assume_role_provider.clear()


class _StubSTSClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failure: Exception | None = None
        self.region_name: str | None = None
        self.expires_on = int(
            (datetime.now(UTC) + timedelta(hours=1)).timestamp()
        )

    def assume_role(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)

        if self.failure is not None:
            raise self.failure

        # Every exchange returns distinct credentials, so that a cached set
        # can be told apart from a refreshed one.
        nth = len(self.calls)
        return {
            "Credentials": {
                "AccessKeyId": f"sts-key-{nth}",
                "SecretAccessKey": f"sts-secret-{nth}",
                "SessionToken": f"sts-token-{nth}",
                "Expiration": datetime.fromtimestamp(self.expires_on, UTC),
            }
        }

    def close(self) -> None:
        pass


@pytest.fixture
def sts_client(monkeypatch: pytest.MonkeyPatch) -> _StubSTSClient:
    client = _StubSTSClient()

    class _StubSession:
        def client(
            self, service_name: str, *, region_name: str
        ) -> _StubSTSClient:
            client.region_name = region_name
            return client

    monkeypatch.setattr(auth.boto3, "Session", _StubSession)
    return client


def _aws_creds(
    access_key_id: str | None = None,
    secret_access_key: str | None = None,
    session_token: str | None = None,
) -> OpenAICreds:
    return {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
        "aws_session_token": session_token,
    }


async def _get_credentials(
    extra_data: dict[str, Any] | None = None,
) -> OpenAICreds:
    headers = (
        {}
        if extra_data is None
        else {UPSTREAM_EXTRA_DATA_HEADER: json.dumps(extra_data)}
    )
    return await auth.get_credentials(
        headers, vendor=Vendor.AWS, endpoint=_ENDPOINT
    )


@pytest.mark.parametrize(
    "extra_data, env, expected",
    [
        pytest.param(None, {}, _aws_creds(), id="nothing_configured"),
        pytest.param(
            {
                "aws_access_key_id": "header-key",
                "aws_secret_access_key": "header-secret",
                "aws_session_token": "header-token",
            },
            {},
            _aws_creds("header-key", "header-secret", "header-token"),
            id="static_credentials_from_upstream_config",
        ),
        pytest.param(
            None,
            {
                "AWS_ACCESS_KEY_ID": "env-key",
                "AWS_SECRET_ACCESS_KEY": "env-secret",
                "AWS_SESSION_TOKEN": "env-token",
            },
            _aws_creds("env-key", "env-secret", "env-token"),
            id="static_credentials_from_environment",
        ),
        pytest.param(
            {
                "aws_access_key_id": "header-key",
                "aws_secret_access_key": "header-secret",
            },
            {
                "AWS_ACCESS_KEY_ID": "env-key",
                "AWS_SECRET_ACCESS_KEY": "env-secret",
                "AWS_SESSION_TOKEN": "env-token",
            },
            _aws_creds("header-key", "header-secret", "env-token"),
            id="upstream_config_overrides_environment_field_by_field",
        ),
        pytest.param(
            {
                "aws_access_key_id": "header-key",
                "aws_secret_access_key": "header-secret",
            },
            {"AWS_ASSUME_ROLE_ARN": _ROLE_ARN},
            _aws_creds("header-key", "header-secret"),
            id="static_credentials_win_over_assume_role",
        ),
    ],
)
async def test_credential_resolution(
    monkeypatch: pytest.MonkeyPatch,
    extra_data: dict[str, Any] | None,
    env: dict[str, str],
    expected: OpenAICreds,
):
    for env_var, value in env.items():
        monkeypatch.setenv(env_var, value)

    assert await _get_credentials(extra_data) == expected


@pytest.mark.parametrize(
    "extra_data",
    [
        pytest.param({"aws_access_key_id": "key"}, id="access_key_id_alone"),
        pytest.param(
            {"aws_secret_access_key": "secret"}, id="secret_access_key_alone"
        ),
        pytest.param({"aws_session_token": "token"}, id="session_token_alone"),
    ],
)
async def test_incomplete_static_credentials_are_rejected(
    extra_data: dict[str, Any],
):
    with pytest.raises(DialException) as exc_info:
        await _get_credentials(extra_data)

    error = exc_info.value
    assert error.status_code == 500
    assert error.message.startswith("Incomplete AWS credentials")


async def test_assume_role_exchanges_the_arn_for_temporary_credentials(
    sts_client: _StubSTSClient,
):
    creds = await _get_credentials({"aws_assume_role_arn": _ROLE_ARN})

    assert creds == _aws_creds("sts-key-1", "sts-secret-1", "sts-token-1")
    assert sts_client.region_name == _REGION
    assert sts_client.calls == [
        {"RoleArn": _ROLE_ARN, "RoleSessionName": "BedrockAccessSession"}
    ]


async def test_assumed_credentials_are_reused_until_they_near_expiration(
    monkeypatch: pytest.MonkeyPatch, sts_client: _StubSTSClient
):
    extra_data = {"aws_assume_role_arn": _ROLE_ARN}

    credentials = await _get_credentials(extra_data)

    assert await _get_credentials(extra_data) == credentials
    assert len(sts_client.calls) == 1

    monkeypatch.setattr(
        auth.time,
        "time",
        lambda: sts_client.expires_on - auth.AWS_EXPIRATION_WINDOW_IN_SEC / 2,
    )

    assert await _get_credentials(extra_data) == _aws_creds(
        "sts-key-2", "sts-secret-2", "sts-token-2"
    )
    assert len(sts_client.calls) == 2


async def test_assume_role_failure_is_reported_as_a_dial_error(
    sts_client: _StubSTSClient,
):
    sts_client.failure = RuntimeError("AccessDenied")

    with pytest.raises(DialException) as exc_info:
        await _get_credentials({"aws_assume_role_arn": _ROLE_ARN})

    error = exc_info.value
    assert error.status_code == 500
    # The role ARN goes to the logs, not to the client.
    assert error.message == "Failed to assume the configured AWS role"
