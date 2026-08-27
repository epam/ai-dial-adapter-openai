import os
from enum import StrEnum
from typing import NamedTuple, assert_never

import pytest

from tests.conftest import AzureOpenAIClientFactory
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.fixtures import maybe_parametrized_fixture
from tests.utils.openai import chat_completion, user

D = DeploymentConfig

# Matches both the bedrock-mantle and the bedrock-runtime hosts.
_BEDROCK_HOST_MARKER = "//bedrock-"

_bedrock_text_deployments: list[D] = [
    deployment
    for deployment in TEST_DEPLOYMENTS_CONFIG.chat_deployments
    if _BEDROCK_HOST_MARKER in deployment.upstream_endpoint
    and not deployment.supports_video_generation
    and not deployment.supports_tts
    and not deployment.supports_stt
    and not deployment.model_features.imageGenerationSupported
]

_STATIC_CREDENTIALS_ENV_VARS = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
)
_ASSUME_ROLE_ENV_VAR = "AWS_ASSUME_ROLE_ARN"


class _AuthMode(StrEnum):
    UPSTREAM_KEY = "upstream_key"
    EXTRA_DATA_STATIC = "extra_data_static"
    ENV_STATIC = "env_static"


class _BedrockAuth(NamedTuple):
    upstream_key: str | None = None
    upstream_extra_data: dict[str, str] | None = None


@maybe_parametrized_fixture(
    params=_bedrock_text_deployments,
    ids=lambda d: d.display_config(),
    skip_reason="No Bedrock text deployments were found",
)
def bedrock_deployment(deployment: D) -> D:
    return deployment


def _read_static_credentials() -> dict[str, str] | None:
    credentials = {
        env_var.lower(): value
        for env_var in _STATIC_CREDENTIALS_ENV_VARS
        if (value := os.getenv(env_var))
    }

    if not {"aws_access_key_id", "aws_secret_access_key"} <= credentials.keys():
        return None

    return credentials


@pytest.fixture(params=list(_AuthMode), ids=lambda mode: mode.value)
def bedrock_auth(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    bedrock_deployment: D,
) -> _BedrockAuth:
    mode: _AuthMode = request.param

    static_credentials = _read_static_credentials()

    def require_static_credentials() -> dict[str, str]:
        if static_credentials is None:
            pytest.skip(
                "AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY are not set"
            )
        return static_credentials

    match mode:
        case _AuthMode.UPSTREAM_KEY:
            if bedrock_deployment.upstream_api_key is None:
                pytest.skip("The deployment has no upstream API key configured")

            return _BedrockAuth(
                upstream_key=bedrock_deployment.upstream_api_key
            )

        case _AuthMode.EXTRA_DATA_STATIC:
            credentials = require_static_credentials()

            # Clearing the environment proves the header is the credential
            # source: otherwise an ignored header would fall back to it, or to
            # an assumed role, and the request would succeed regardless.
            for env_var in (
                *_STATIC_CREDENTIALS_ENV_VARS,
                _ASSUME_ROLE_ENV_VAR,
            ):
                monkeypatch.delenv(env_var, raising=False)

            return _BedrockAuth(upstream_extra_data=credentials)

        case _AuthMode.ENV_STATIC:
            require_static_credentials()

            return _BedrockAuth()

        case _:
            assert_never(mode)


async def test_bedrock_text(
    create_azure_openai_client: AzureOpenAIClientFactory,
    bedrock_deployment: D,
    bedrock_auth: _BedrockAuth,
):
    response = await chat_completion(
        create_azure_openai_client(
            bedrock_deployment.id_,
            upstream_endpoint=bedrock_deployment.upstream_endpoint,
            upstream_key=bedrock_auth.upstream_key,
            upstream_extra_data=bedrock_auth.upstream_extra_data,
        ),
        stream=False,
        deployment_id=bedrock_deployment.model_name,
        messages=[
            user("2+2?"),
        ],
    )

    assert "4" in response.content
    assert response.response.choices[0].finish_reason == "stop"
