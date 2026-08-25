import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any, assert_never

import boto3
from aidial_sdk.exceptions import HTTPException as DialException
from aidial_sdk.exceptions import InternalServerError
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity.aio import DefaultAzureCredential
from pydantic import BaseModel
from typing_extensions import TypedDict

from aidial_adapter_openai.configuration.app_config import (
    DeploymentAPIEndpoint,
    Vendor,
)
from aidial_adapter_openai.utils.cache import cache
from aidial_adapter_openai.utils.concurrency import run_in_threadpool
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import BedrockOpenAIEndpoint
from aidial_adapter_openai.utils.upstream_headers import (
    UpstreamExtraData,
    get_upstream_extra_data,
)

EXPIRATION_WINDOW_IN_SEC: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRATION_WINDOW", 10)
)
AZURE_OPEN_AI_SCOPE: str = os.getenv(
    "AZURE_OPEN_AI_SCOPE", "https://cognitiveservices.azure.com/.default"
)
_ASSUME_ROLE_SESSION_NAME = "BedrockAccessSession"


class _AzureTokenProvider:
    _credential: DefaultAzureCredential
    _access_token: AccessToken | None

    def __init__(self) -> None:
        self._credential = DefaultAzureCredential()
        self._access_token = None

    async def aclose(self):
        await self._credential.close()

    async def get_token(self) -> str:
        now = int(time.time())

        if (
            self._access_token is None
            or now + EXPIRATION_WINDOW_IN_SEC > self._access_token.expires_on
        ):
            try:
                self._access_token = await self._credential.get_token(
                    AZURE_OPEN_AI_SCOPE
                )
                logger.debug(
                    f"Obtained new Azure access token, expires on {self._access_token.expires_on}"
                )
            except ClientAuthenticationError as e:
                logger.error(
                    f"Default Azure credential failed with the error: {e.message}"
                )
                raise DialException(
                    "Authentication failed", 401, "Unauthorized"
                )

        return self._access_token.token


async def _close_token_provider(provider: _AzureTokenProvider):
    await provider.aclose()


@cache(_close_token_provider)
def get_azure_token_provider() -> _AzureTokenProvider:
    return _AzureTokenProvider()


class AWSCredentials(BaseModel):
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None


class _AWSAssumeRoleProvider:
    """
    Exchanges the ambient adapter credentials for temporary credentials of
    the given role and reuses them until they are about to expire.
    """

    def __init__(self, role_arn: str, region: str) -> None:
        self._role_arn = role_arn
        self._region = region
        self._sts_client: Any = None
        self._credentials: AWSCredentials | None = None
        self._expires_on: int = 0
        self._lock = asyncio.Lock()

    def close(self) -> None:
        if self._sts_client is not None:
            self._sts_client.close()
            self._sts_client = None

    def _assume_role(self) -> tuple[AWSCredentials, int]:
        if self._sts_client is None:
            self._sts_client = boto3.Session().client(
                "sts", region_name=self._region
            )

        creds = self._sts_client.assume_role(
            RoleArn=self._role_arn,
            RoleSessionName=_ASSUME_ROLE_SESSION_NAME,
        )["Credentials"]

        return (
            AWSCredentials(
                aws_access_key_id=creds["AccessKeyId"],
                aws_secret_access_key=creds["SecretAccessKey"],
                aws_session_token=creds["SessionToken"],
            ),
            int(creds["Expiration"].timestamp()),
        )

    async def get_credentials(self) -> AWSCredentials:
        async with self._lock:
            now = int(time.time())

            if (
                self._credentials is None
                or now + EXPIRATION_WINDOW_IN_SEC > self._expires_on
            ):
                try:
                    (
                        self._credentials,
                        self._expires_on,
                    ) = await run_in_threadpool(self._assume_role)
                except Exception as e:
                    logger.error(f"Assuming role {self._role_arn} failed: {e}")
                    raise InternalServerError(
                        "Failed to assume the configured AWS role"
                    ) from e

                logger.debug(
                    f"Assumed role {self._role_arn}, "
                    f"credentials expire on {self._expires_on}"
                )

            return self._credentials


async def _close_assume_role_provider(provider: _AWSAssumeRoleProvider) -> None:
    await run_in_threadpool(provider.close)


@cache(_close_assume_role_provider)
def get_assume_role_provider(
    role_arn: str, region: str
) -> _AWSAssumeRoleProvider:
    return _AWSAssumeRoleProvider(role_arn, region)


class AWSClientCredentials(BaseModel):
    aws_access_key_id: str
    aws_secret_access_key: str
    aws_session_token: str | None = None

    async def get_credentials(self, region: str) -> AWSCredentials:
        return AWSCredentials(
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            aws_session_token=self.aws_session_token,
        )


class AWSAssumeRoleCredentials(BaseModel):
    aws_assume_role_arn: str

    async def get_credentials(self, region: str) -> AWSCredentials:
        return await get_assume_role_provider(
            self.aws_assume_role_arn, region
        ).get_credentials()


def _select_credentials(
    extra_data: UpstreamExtraData,
) -> AWSClientCredentials | AWSAssumeRoleCredentials | None:
    access_key_id = extra_data.aws_access_key_id or os.getenv(
        "AWS_ACCESS_KEY_ID"
    )
    secret_access_key = extra_data.aws_secret_access_key or os.getenv(
        "AWS_SECRET_ACCESS_KEY"
    )
    session_token = extra_data.aws_session_token or os.getenv(
        "AWS_SESSION_TOKEN"
    )
    assume_role_arn = extra_data.aws_assume_role_arn or os.getenv(
        "AWS_ASSUME_ROLE_ARN"
    )

    if access_key_id and secret_access_key:
        return AWSClientCredentials(
            aws_access_key_id=access_key_id,
            aws_secret_access_key=secret_access_key,
            aws_session_token=session_token,
        )

    if access_key_id or secret_access_key:
        raise InternalServerError(
            "Incomplete AWS credentials: aws_access_key_id and "
            "aws_secret_access_key must be configured together."
        )

    if session_token:
        raise InternalServerError(
            "Incomplete AWS credentials: aws_session_token requires "
            "aws_access_key_id and aws_secret_access_key."
        )

    if assume_role_arn:
        return AWSAssumeRoleCredentials(aws_assume_role_arn=assume_role_arn)

    return None


class AWSCloudUpstreamConfig(BaseModel):
    credentials: AWSClientCredentials | AWSAssumeRoleCredentials | None = None

    @classmethod
    def create(cls, headers: Mapping[str, str]) -> "AWSCloudUpstreamConfig":
        extra_data = get_upstream_extra_data(headers)
        return cls(credentials=_select_credentials(extra_data))

    async def get_credentials(self, aws_region: str) -> AWSCredentials:
        if self.credentials is None:
            return AWSCredentials()
        return await self.credentials.get_credentials(aws_region)


class OpenAICreds(TypedDict, total=False):
    api_key: str
    azure_ad_token: str

    aws_access_key_id: str | None
    aws_secret_access_key: str | None
    aws_session_token: str | None


async def get_credentials(
    request_headers: Mapping[str, str],
    *,
    vendor: Vendor,
    endpoint: DeploymentAPIEndpoint | None,
) -> OpenAICreds:
    api_key = request_headers.get("X-UPSTREAM-KEY")
    if api_key is not None:
        return {"api_key": api_key}

    match vendor:
        case Vendor.AZURE:
            return {
                "azure_ad_token": await get_azure_token_provider().get_token()
            }
        case Vendor.AWS:
            if not isinstance(endpoint, BedrockOpenAIEndpoint):
                raise InternalServerError(
                    "Unexpected endpoint for the AWS vendor: "
                    f"{type(endpoint).__name__}"
                )

            upstream_config = AWSCloudUpstreamConfig.create(request_headers)
            creds = await upstream_config.get_credentials(
                endpoint.bedrock_region
            )

            return {
                "aws_access_key_id": creds.aws_access_key_id,
                "aws_secret_access_key": creds.aws_secret_access_key,
                "aws_session_token": creds.aws_session_token,
            }
        case Vendor.VLLM | Vendor.OPENAI_PLATFORM | Vendor.ALIBABA:
            raise DialException(
                "X-UPSTREAM-KEY header is missing", 401, "Unauthorized"
            )
        case _:
            assert_never(vendor)
