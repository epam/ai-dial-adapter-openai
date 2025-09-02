from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Iterator, List

from openai import NOT_GIVEN, NotGiven
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletionMessageParam,
    ChatCompletionToolParam,
)
from openai.types.chat.completion_create_params import Function, ResponseFormat

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
    max_completion_tokens: int | NotGiven
    stop: List[str] | NotGiven

    n: int | NotGiven

    functions: List[Function] | NotGiven
    tools: List[ChatCompletionToolParam] | NotGiven
    temperature: float | NotGiven

    reasoning_effort: ReasoningEffort | NotGiven
    response_format: ResponseFormat | NotGiven

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
        *,
        name: str,
        messages: List[ChatCompletionMessageParam],
        max_tokens: int | NotGiven = NOT_GIVEN,
        max_completion_tokens: int | NotGiven = NOT_GIVEN,
        stop: List[str] | NotGiven = NOT_GIVEN,
        n: int | NotGiven = NOT_GIVEN,
        functions: List[Function] | NotGiven = NOT_GIVEN,
        tools: List[ChatCompletionToolParam] | NotGiven = NOT_GIVEN,
        temperature: float | NotGiven = NOT_GIVEN,
        reasoning_effort: ReasoningEffort | NotGiven = NOT_GIVEN,
        response_format: ResponseFormat | NotGiven = NOT_GIVEN,
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
                max_tokens=max_tokens,
                max_completion_tokens=max_completion_tokens,
                stop=stop,
                n=n,
                functions=functions,
                tools=tools,
                temperature=temperature,
                reasoning_effort=reasoning_effort,
                response_format=response_format,
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
    def supports_reasoning_summary(self):
        return self.deployment_config.model_features.reasoningSummarySupported

    @property
    def supports_function_calling(self):
        return self.deployment_config.model_features.toolsSupported

    @property
    def supports_parallel_function_calling(self):
        return self.deployment_config.model_features.parallelToolCallsSupported

    @property
    def supports_stop(self):
        return self.deployment_config.model_features.stopSupported

    @property
    def supports_temperature(self):
        return self.deployment_config.model_features.temperatureSupported

    @property
    def supports_response_format_json_object(self):
        return (
            self.deployment_config.model_features.responseFormatJsonObjectSupported
        )

    @property
    def supports_response_format_json_schema(self):
        return (
            self.deployment_config.model_features.responseFormatJsonSchemaSupported
        )
