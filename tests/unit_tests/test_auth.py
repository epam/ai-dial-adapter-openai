import json
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
import respx
from aidial_sdk.exceptions import HTTPException as DialException

from aidial_adapter_openai.configuration.app_config import (
    Vendor,
)
from aidial_adapter_openai.utils import auth, session_tags
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import BedrockOpenAIEndpoint
from aidial_adapter_openai.utils.upstream_headers import (
    _UPSTREAM_EXTRA_DATA_HEADER,
)

_REGION = "us-east-1"
_ENDPOINTS = [
    BedrockOpenAIEndpoint(
        bedrock_region=_REGION,
        client="mantle",
        openai_base_url=f"https://bedrock-mantle.{_REGION}.api.aws/openai/v1",
    ),
    BedrockOpenAIEndpoint(
        bedrock_region=_REGION,
        client="runtime",
        openai_base_url=f"https://bedrock-runtime.{_REGION}.amazonaws.com/openai/v1",
    ),
]
_ROLE_ARN = "arn:aws:iam::123456789012:role/BedrockAccess"
_DIAL_URL = "http://test-dial-url"
_AWS_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_ASSUME_ROLE_ARN",
    "AWS_SESSION_TAGS_FIELDS",
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


@pytest.fixture(params=_ENDPOINTS, ids=lambda endpoint: endpoint.client)
def endpoint(request: pytest.FixtureRequest) -> BedrockOpenAIEndpoint:
    """
    Both Bedrock clients resolve the credentials the same way: only the region
    of the endpoint takes part in it.
    """
    return request.param


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
    endpoint: BedrockOpenAIEndpoint,
    extra_data: dict[str, Any] | None = None,
    api_key: str | None = None,
) -> OpenAICreds:
    headers = (
        {}
        if extra_data is None
        else {_UPSTREAM_EXTRA_DATA_HEADER: json.dumps(extra_data)}
    )
    if api_key is not None:
        headers["api-key"] = api_key
    return await auth.get_credentials(
        headers, vendor=Vendor.AWS, endpoint=endpoint
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
    endpoint: BedrockOpenAIEndpoint,
    extra_data: dict[str, Any] | None,
    env: dict[str, str],
    expected: OpenAICreds,
):
    for env_var, value in env.items():
        monkeypatch.setenv(env_var, value)

    assert await _get_credentials(endpoint, extra_data) == expected


@pytest.mark.parametrize(
    "extra_data, err_msg",
    [
        pytest.param(
            {"aws_access_key_id": "key"},
            "Incomplete AWS credentials: aws_access_key_id and aws_secret_access_key must be configured together.",
            id="access_key_id_alone",
        ),
        pytest.param(
            {"aws_secret_access_key": "secret"},
            "Incomplete AWS credentials: aws_access_key_id and aws_secret_access_key must be configured together.",
            id="secret_access_key_alone",
        ),
        pytest.param(
            {"aws_session_token": "token"},
            "Incomplete AWS credentials: aws_session_token requires aws_access_key_id and aws_secret_access_key.",
            id="session_token_alone",
        ),
    ],
)
async def test_incomplete_static_credentials_are_rejected(
    endpoint: BedrockOpenAIEndpoint, extra_data: dict[str, Any], err_msg: str
):
    with pytest.raises(DialException) as exc_info:
        await _get_credentials(endpoint, extra_data)

    error = exc_info.value
    assert error.status_code == 500
    assert error.message == err_msg


async def test_assume_role_exchanges_the_arn_for_temporary_credentials(
    endpoint: BedrockOpenAIEndpoint, sts_client: _StubSTSClient
):
    creds = await _get_credentials(endpoint, {"aws_assume_role_arn": _ROLE_ARN})

    assert creds == _aws_creds("sts-key-1", "sts-secret-1", "sts-token-1")
    assert sts_client.region_name == _REGION
    assert sts_client.calls == [
        {"RoleArn": _ROLE_ARN, "RoleSessionName": "BedrockAccessSession"}
    ]


async def test_assumed_credentials_are_reused_until_they_near_expiration(
    monkeypatch: pytest.MonkeyPatch,
    endpoint: BedrockOpenAIEndpoint,
    sts_client: _StubSTSClient,
):
    extra_data = {"aws_assume_role_arn": _ROLE_ARN}

    credentials = await _get_credentials(endpoint, extra_data)

    assert await _get_credentials(endpoint, extra_data) == credentials
    assert len(sts_client.calls) == 1

    monkeypatch.setattr(
        auth.time,
        "time",
        lambda: sts_client.expires_on - auth.EXPIRATION_WINDOW_IN_SEC / 2,
    )

    assert await _get_credentials(endpoint, extra_data) == _aws_creds(
        "sts-key-2", "sts-secret-2", "sts-token-2"
    )
    assert len(sts_client.calls) == 2


async def test_assume_role_failure_is_reported_as_a_dial_error(
    endpoint: BedrockOpenAIEndpoint, sts_client: _StubSTSClient
):
    sts_client.failure = RuntimeError("AccessDenied")

    with pytest.raises(DialException) as exc_info:
        await _get_credentials(endpoint, {"aws_assume_role_arn": _ROLE_ARN})

    error = exc_info.value
    assert error.status_code == 500
    # The role ARN goes to the logs, not to the client.
    assert error.message == "Failed to assume the configured AWS role"


class TestSessionTags:
    """
    The session tags are only honoured by the assume role call, so the other
    credential sources must neither request nor pass them.
    """

    @pytest.fixture(autouse=True)
    def session_tags_configured(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(session_tags, "DIAL_URL", _DIAL_URL)
        monkeypatch.setenv("AWS_SESSION_TAGS_FIELDS", "roles.0")

    @pytest.fixture
    def user_info(self):
        with respx.mock(
            base_url=_DIAL_URL + "/v1",
            assert_all_called=False,
            assert_all_mocked=True,
        ) as router:
            yield router.get("/user/info")

    @staticmethod
    def _respond_with_role(user_info: Any, *roles: str) -> None:
        user_info.side_effect = [
            httpx.Response(200, json={"roles": [role]}) for role in roles
        ]

    async def test_session_tags_are_passed_to_assume_role(
        self,
        endpoint: BedrockOpenAIEndpoint,
        sts_client: _StubSTSClient,
        user_info: Any,
    ):
        self._respond_with_role(user_info, "admin")

        await _get_credentials(
            endpoint, {"aws_assume_role_arn": _ROLE_ARN}, api_key="key-1"
        )

        assert sts_client.calls == [
            {
                "RoleArn": _ROLE_ARN,
                "RoleSessionName": "BedrockAccessSession",
                "Tags": [{"Key": "roles.0", "Value": "admin"}],
            }
        ]

    async def test_assumed_credentials_are_not_shared_between_tag_sets(
        self,
        endpoint: BedrockOpenAIEndpoint,
        sts_client: _StubSTSClient,
        user_info: Any,
    ):
        self._respond_with_role(user_info, "admin", "guest", "admin")
        extra_data = {"aws_assume_role_arn": _ROLE_ARN}

        admin = await _get_credentials(endpoint, extra_data, api_key="key-1")
        guest = await _get_credentials(endpoint, extra_data, api_key="key-2")

        assert admin != guest
        assert len(sts_client.calls) == 2

        # The credentials of the first tag set are reused, not re-exchanged.
        assert (
            await _get_credentials(endpoint, extra_data, api_key="key-3")
            == admin
        )
        assert len(sts_client.calls) == 2

    async def test_static_credentials_dont_resolve_session_tags(
        self, endpoint: BedrockOpenAIEndpoint, user_info: Any
    ):
        creds = await _get_credentials(
            endpoint,
            {
                "aws_access_key_id": "header-key",
                "aws_secret_access_key": "header-secret",
            },
            api_key="key-1",
        )

        assert creds == _aws_creds("header-key", "header-secret")
        assert not user_info.called
