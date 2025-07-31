from __future__ import annotations

import functools
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


def exclude_deployments(
    deployment_types: List[ChatCompletionDeploymentType],
):
    def wrapper(func: TestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: TestSuite):
            if s.deployment_type in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper


def include_deployments(
    deployment_types: List[ChatCompletionDeploymentType],
):
    def wrapper(func: TestSuiteBuilder):
        @functools.wraps(func)
        def wrapped(s: TestSuite):
            if s.deployment_type not in deployment_types:
                return
            return func(s)

        return wrapped

    return wrapper
