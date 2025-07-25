from __future__ import annotations

import functools
import json
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    Iterator,
    List,
    Literal,
    TypeVar,
    assert_never,
)

from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from tests.utils.openai import ChatCompletionResult, ExpectedException


class UpstreamConfig(ExtraAllowedModel):
    endpoint: str
    key: str


class ModelConfig(ExtraAllowedModel):
    type: Literal["chat", "embedding"]
    overrideName: str | None = None
    upstreams: List[UpstreamConfig]


class CoreConfig(ExtraAllowedModel):
    models: Dict[str, ModelConfig]

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            test_config = json.load(f)

        return cls(**test_config)


class EmbeddingsDeploymentType(BaseModel):
    pass


_T = TypeVar("_T")

DeploymentType = ChatCompletionDeploymentType | EmbeddingsDeploymentType


def _classify_deployment(
    app_config: ApplicationConfig,
    model_config: ModelConfig,
    deployment_id: str,
    upstream_endpoint: str,
) -> DeploymentType:
    if model_config.type == "chat":
        return app_config.get_chat_completion_deployment_type(
            deployment_id, upstream_endpoint
        ).deployment_type
    elif model_config.type == "embedding":
        return EmbeddingsDeploymentType()
    else:
        assert_never(model_config.type)


class DeploymentConfig(BaseModel, Generic[_T]):
    id_: str
    type_: _T

    model_name: str
    upstream_endpoint: str
    upstream_api_key: str
    upstream_idx: int | None

    @property
    def upstream_headers(self) -> Dict[str, str]:
        return {
            "X-UPSTREAM-KEY": self.upstream_api_key,
            "X-UPSTREAM-ENDPOINT": self.upstream_endpoint,
        }

    @classmethod
    def create_deployments(
        cls,
        core_config: CoreConfig,
        get_deployment_type: Callable[[ModelConfig, str, str], _T],
    ) -> List[DeploymentConfig[_T]]:
        configs = []
        for deployment_id, model_config in core_config.models.items():
            for upstream_index, upstream_config in enumerate(
                model_config.upstreams
            ):
                upstream_endpoint = upstream_config.endpoint

                upstream_idx = (
                    None if len(model_config.upstreams) <= 1 else upstream_index
                )

                configs.append(
                    cls(
                        upstream_idx=upstream_idx,
                        id_=deployment_id,
                        model_name=model_config.overrideName or deployment_id,
                        type_=get_deployment_type(
                            model_config, deployment_id, upstream_endpoint
                        ),
                        upstream_endpoint=upstream_endpoint,
                        upstream_api_key=upstream_config.key,
                    )
                )
        return configs


class TestDeployments(BaseModel):
    __test__ = False

    deployments: List[DeploymentConfig[DeploymentType]]
    app_config: ApplicationConfig

    @classmethod
    def from_config(cls, config_path: str):
        app_config = ApplicationConfig.from_env()
        core_config = CoreConfig.from_config(config_path)

        return cls(
            app_config=app_config,
            deployments=DeploymentConfig.create_deployments(
                core_config,
                functools.partial(_classify_deployment, app_config),
            ),
        )

    @property
    def chat_deployments(
        self,
    ) -> Generator[DeploymentConfig[ChatCompletionDeploymentType], None, None]:
        for deployment in self.deployments:
            if isinstance(deployment.type_, ChatCompletionDeploymentType):
                yield deployment  # type: ignore

    @property
    def embedding_deployment(
        self,
    ) -> Generator[DeploymentConfig[EmbeddingsDeploymentType], None, None]:
        for deployment in self.deployments:
            if isinstance(deployment.type_, EmbeddingsDeploymentType):
                yield deployment  # type: ignore


def sanitize_id_part(value: Any) -> str:
    """Convert any value to a pytest-safe identifier part."""
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, (int, float)):
        return str(value).replace(".", "p")  # e.g., 0.5 -> 0p5
    if value is None:
        return "none"

    value_str = str(value)
    sanitized = "".join(c if c.isalnum() else "_" for c in value_str)
    return sanitized.strip("_")


@dataclass
class ChatTestCase:
    __test__ = False

    deployment_config: DeploymentConfig[ChatCompletionDeploymentType]

    name: str
    streaming: bool

    messages: List[ChatCompletionMessageParam]

    expected: Callable[[ChatCompletionResult], bool] | ExpectedException

    max_tokens: int | NotGiven
    stop: List[str] | NotGiven

    n: int | NotGiven

    functions: List[Function] | NotGiven
    tools: List[ChatCompletionToolParam] | NotGiven
    temperature: float | NotGiven

    def get_id(self):
        upstream_idx = self.deployment_config.upstream_idx
        parts = [
            sanitize_id_part(self.name),
            sanitize_id_part(self.deployment_config.type_.value),
            sanitize_id_part(self.deployment_config.id_),
            *([] if upstream_idx is None else [f"upstream:{upstream_idx}"]),
            f"stream:{sanitize_id_part(self.streaming)}",
        ]

        return "/".join(parts)


ChatTestSuiteBuilder = Callable[["ChatTestSuite"], None]


@dataclass
class ChatTestSuite:
    __test__ = False

    deployment_config: DeploymentConfig[ChatCompletionDeploymentType]
    streaming: bool
    test_cases: List[ChatTestCase] = field(default_factory=list)

    def test_case(
        self,
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            Callable[[ChatCompletionResult], bool] | ExpectedException
        ) = lambda *args, **kwargs: True,
        **kwargs,
    ) -> ChatTestSuite:
        self.test_cases.append(
            ChatTestCase(
                deployment_config=self.deployment_config,
                name=name,
                streaming=self.streaming,
                messages=messages,
                expected=expected,
                max_tokens=kwargs.get("max_tokens") or NOT_GIVEN,
                stop=kwargs.get("stop") or NOT_GIVEN,
                n=kwargs.get("n") or NOT_GIVEN,
                functions=kwargs.get("functions") or NOT_GIVEN,
                tools=kwargs.get("tools") or NOT_GIVEN,
                temperature=kwargs.get("temperature") or NOT_GIVEN,
            )
        )
        return self

    def __iter__(self) -> Iterator[ChatTestCase]:
        return iter(self.test_cases)

    def __len__(self):
        return len(self.test_cases)

    @property
    def deployment_type(self) -> ChatCompletionDeploymentType:
        return self.deployment_config.type_

    @classmethod
    def create(
        cls,
        deployment_config: DeploymentConfig[ChatCompletionDeploymentType],
        streaming: bool,
        case_builder: ChatTestSuiteBuilder,
    ) -> "ChatTestSuite":
        suite = cls(deployment_config, streaming)
        case_builder(suite)
        return suite


def exclude_chat_deployments(
    deployment_types: List[ChatCompletionDeploymentType],
):
    def wrapper(func: ChatTestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: ChatTestSuite):
            if s.deployment_type in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper


def include_chat_deployments(
    deployment_types: List[ChatCompletionDeploymentType],
):
    def wrapper(func: ChatTestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: ChatTestSuite):
            if s.deployment_type not in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper
