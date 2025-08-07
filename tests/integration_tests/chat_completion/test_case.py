from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, List

from openai import NOT_GIVEN, NotGiven
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.base import DeploymentConfig, sanitize_id_part
from tests.utils.openai import ChatCompletionResult, ExpectedException


@dataclass
class TestCase:
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

    extra_body: dict | None

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


TestSuiteBuilder = Callable[["TestSuite"], None]


@dataclass
class TestSuite:
    __test__ = False

    deployment_config: DeploymentConfig[ChatCompletionDeploymentType]
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
    ) -> TestSuite:
        self.test_cases.append(
            TestCase(
                deployment_config=self.deployment_config,
                name=name,
                streaming=self.streaming,
                messages=messages,
                expected=expected,
                max_tokens=kwargs.pop("max_tokens", None) or NOT_GIVEN,
                stop=kwargs.pop("stop", None) or NOT_GIVEN,
                n=kwargs.pop("n", None) or NOT_GIVEN,
                functions=kwargs.pop("functions", None) or NOT_GIVEN,
                tools=kwargs.pop("tools", None) or NOT_GIVEN,
                temperature=kwargs.pop("temperature", None) or NOT_GIVEN,
                extra_body=kwargs,
            )
        )
        return self

    def __iter__(self) -> Iterator[TestCase]:
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
        case_builder: TestSuiteBuilder,
    ) -> TestSuite:
        suite = cls(deployment_config, streaming)
        case_builder(suite)
        return suite

    @property
    def supports_system_prompt(self):
        return self.deployment_config.model_features.systemPromptSupported

    @property
    def supports_vision(self):
        types = self.deployment_config.model_attachments or []
        return any(
            ty.startswith("image/") or ty.startswith("*/") for ty in types
        )

    @property
    def supports_reasoning(self):
        return self.deployment_config.model_features.reasoningSupported

    @property
    def supports_function_calling(self):
        return self.deployment_config.model_features.toolsSupported

    @property
    def supports_parallel_function_calling(self):
        return self.deployment_config.model_features.parallelToolCallsSupported

    @property
    def supports_stop(self):
        return self.deployment_config.model_features.stopSupported
