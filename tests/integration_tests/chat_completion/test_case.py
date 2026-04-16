from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field

from openai import Omit, omit
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

    messages: list[ChatCompletionMessageParam]

    expected: Callable[[ChatCompletionResult], bool] | ExpectedException

    max_tokens: int | Omit
    max_completion_tokens: int | Omit
    stop: list[str] | Omit

    n: int | Omit

    functions: list[Function] | Omit
    tools: list[ChatCompletionToolParam] | Omit
    temperature: float | Omit

    reasoning_effort: ReasoningEffort | Omit
    response_format: ResponseFormat | Omit

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
    test_cases: list[TestCase] = field(default_factory=list)

    def test_case(
        self,
        *,
        name: str,
        messages: list[ChatCompletionMessageParam],
        max_tokens: int | Omit = omit,
        max_completion_tokens: int | Omit = omit,
        stop: list[str] | Omit = omit,
        n: int | Omit = omit,
        functions: list[Function] | Omit = omit,
        tools: list[ChatCompletionToolParam] | Omit = omit,
        temperature: float | Omit = omit,
        reasoning_effort: ReasoningEffort | Omit = omit,
        response_format: ResponseFormat | Omit = omit,
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
        return self.deployment_config.supports_vision

    @property
    def supports_image_generation(self):
        return self.deployment_config.model_features.imageGenerationSupported

    @property
    def supports_image_editing(self):
        return self.deployment_config.model_features.imageEditingSupported

    @property
    def supports_reasoning(self):
        return self.deployment_config.model_features.reasoningSupported

    @property
    def supports_empty_dialog(self):
        return self.deployment_config.model_features.emptyDialogSupported

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
        return self.deployment_config.model_features.responseFormatJsonObjectSupported

    @property
    def supports_response_format_json_schema(self):
        return self.deployment_config.model_features.responseFormatJsonSchemaSupported
