import logging
import re
from typing import Generator, List

import openai
import pytest

from tests.integration_tests.chat_completion.file_input import (
    build_file_input_common,
)
from tests.integration_tests.chat_completion.test_case import (
    TestCase,
    TestSuite,
    TestSuiteBuilder,
)
from tests.integration_tests.chat_completion.text import (
    build_multi_system,
    build_stop_sequence,
    build_text_common,
)
from tests.integration_tests.chat_completion.tools import build_tools_common
from tests.integration_tests.chat_completion.vision import build_vision_common
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.openai import (
    ChatCompletionResult,
    ExpectedException,
    chat_completion,
)

logger = logging.getLogger(__name__)


def create_test_cases(
    builders: List[TestSuiteBuilder],
) -> Generator[TestCase, None, None]:
    for streaming in (False, True):
        for deployment in TEST_DEPLOYMENTS_CONFIG.chat_deployments:
            suite = TestSuite(deployment, streaming)
            for builder in builders:
                builder(suite)
            yield from suite


@pytest.mark.parametrize(
    "test_case",
    create_test_cases(
        [
            build_text_common,
            build_stop_sequence,
            build_multi_system,
            build_tools_common,
            build_vision_common,
            build_file_input_common,
        ]
    ),
    ids=lambda tc: tc.get_id() if isinstance(tc, TestCase) else "na",
)
async def test_chat_completion(test_case: TestCase, create_openai_client):
    client: openai.AsyncAzureOpenAI = create_openai_client(
        test_case.deployment_config
    )

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(
            client,
            test_case.deployment_config.model_name,
            test_case.messages,
            test_case.streaming,
            stop=test_case.stop,
            max_tokens=test_case.max_tokens,
            max_completion_tokens=test_case.max_completion_tokens,
            n=test_case.n,
            functions=test_case.functions,
            tools=test_case.tools,
            temperature=test_case.temperature,
            reasoning_effort=test_case.reasoning_effort,
            extra_body=test_case.extra_body,
        )

    if isinstance(test_case.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        expected = test_case.expected
        assert isinstance(actual_exc, expected.type)

        if (status_code := expected.status_code) is not None:
            actual_status_code = getattr(actual_exc, "status_code", None)
            assert actual_status_code == status_code

        if (message := expected.message) is not None:
            assert re.search(message, str(actual_exc))

        if (display_message := expected.display_message) is not None:
            assert re.search(display_message, str(actual_exc))
    else:
        actual_output = await run_chat_completion()
        assert test_case.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
