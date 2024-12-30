import functools
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Self

from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function
from pydantic import BaseModel

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.constant import ChatCompletionDeploymentType
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from tests.utils.openai import ChatCompletionResult, ExpectedException


class UpstreamConfig(ExtraAllowedModel):
    endpoint: str
    key: str


class ModelConfig(ExtraAllowedModel):
    upstreams: List[UpstreamConfig]


class CoreConfig(ExtraAllowedModel):
    models: Dict[str, ModelConfig]

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, "r") as f:
            test_config = json.load(f)

        return cls(**test_config)


class DeploymentConfig(BaseModel):
    test_deployment_id: str

    deployment_id: str
    deployment_type: ChatCompletionDeploymentType
    upstream_endpoint: str
    upstream_api_key: str

    @property
    def upstream_headers(self) -> Dict[str, str]:
        return {
            "X-UPSTREAM-KEY": self.upstream_api_key,
            "X-UPSTREAM-ENDPOINT": self.upstream_endpoint,
        }

    @classmethod
    def create_deployments(
        cls, core_config: CoreConfig, app_config: ApplicationConfig
    ) -> List[Self]:
        configs = []
        for model_name, model_config in core_config.models.items():
            deployment_type = app_config.get_chat_completion_deployment_type(
                model_name
            )
            for upstream_index, upstream_config in enumerate(
                model_config.upstreams
            ):
                test_deployment_id = f"{deployment_type.value}__{model_name}"
                if len(model_config.upstreams) > 1:
                    test_deployment_id += f"_upstream_{upstream_index}"
                configs.append(
                    cls(
                        test_deployment_id=test_deployment_id,
                        deployment_id=model_name,
                        deployment_type=deployment_type,
                        upstream_endpoint=upstream_config.endpoint,
                        upstream_api_key=upstream_config.key,
                    )
                )
        return configs


class TestDeployments(BaseModel):
    __test__ = False
    deployments: list[DeploymentConfig]
    app_config: ApplicationConfig

    @classmethod
    def from_config(cls, config_path: str):
        app_config = ApplicationConfig.from_env()

        core_config = CoreConfig.from_config(config_path)

        return cls(
            app_config=app_config,
            deployments=DeploymentConfig.create_deployments(
                core_config, app_config
            ),
        )


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
class TestCase:
    __test__ = False

    deployment_config: DeploymentConfig

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
        parts = [
            sanitize_id_part(self.name),
            f"{sanitize_id_part(self.deployment_config.test_deployment_id)}",
            f"stream:{sanitize_id_part(self.streaming)}",
        ]

        if self.max_tokens:
            parts.append(f"tokens:{sanitize_id_part(self.max_tokens)}")
        if self.stop:
            parts.append(f"stop:{sanitize_id_part(self.stop)}")
        if self.n and self.n != 1:
            parts.append(f"n:{sanitize_id_part(self.n)}")
        if self.temperature and self.temperature != 0.0:
            parts.append(f"temp:{sanitize_id_part(self.temperature)}")

        return "/".join(parts)


TestSuiteBuilder = Callable[["TestSuite"], None]


@dataclass
class TestSuite:
    __test__ = False

    deployment_config: DeploymentConfig
    streaming: bool
    test_cases: List[TestCase] = field(default_factory=list)

    def test_case(
        self,
        name: str,
        messages: List[ChatCompletionMessageParam],
        expected: (
            Callable[[ChatCompletionResult], bool] | ExpectedException
        ) = lambda *args, **kwargs: True,
        **kwargs,
    ) -> "TestSuite":
        self.test_cases.append(
            TestCase(
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

    def __iter__(self) -> Iterator[TestCase]:
        return iter(self.test_cases)

    def __len__(self):
        return len(self.test_cases)

    @property
    def deployment_type(self) -> ChatCompletionDeploymentType:
        return self.deployment_config.deployment_type

    @classmethod
    def create(
        cls,
        deployment_config: DeploymentConfig,
        streaming: bool,
        case_builder: TestSuiteBuilder,
    ) -> "TestSuite":
        suite = cls(deployment_config, streaming)
        case_builder(suite)
        return suite


def exclude_deployments(deployment_types: List[ChatCompletionDeploymentType]):
    def wrapper(func: TestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: TestSuite):
            if s.deployment_type in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper


def include_deployments(deployment_types: List[ChatCompletionDeploymentType]):
    def wrapper(func: TestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: TestSuite):
            if s.deployment_type not in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper
