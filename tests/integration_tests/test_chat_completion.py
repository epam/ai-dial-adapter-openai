import re
from typing import List

import pytest

from tests.conftest import TEST_DEPLOYMENTS_CONFIG_PATH
from tests.integration_tests.base import (
    TestCase,
    TestCaseBuilder,
    TestDeployments,
    TestSuite,
)
from tests.integration_tests.chat_completion_cases.text import (
    test_case_builder as text_input_test_case_builder,
)
from tests.integration_tests.chat_completion_cases.tools import (
    test_case_builder as tools_test_case_builder,
)
from tests.utils.openai import (
    ChatCompletionResult,
    ExpectedException,
    chat_completion,
)


def create_test_cases(
    test_case_builder: TestCaseBuilder,
) -> List[TestCase]:

    return [
        test_case
        for streaming in (False, True)
        # for streaming in (True, False)
        for deployment in TestDeployments.from_config(
            TEST_DEPLOYMENTS_CONFIG_PATH
        ).deployments
        for test_case in TestSuite.create(
            deployment, streaming, test_case_builder
        )
    ]


@pytest.mark.parametrize(
    "test_case",
    [
        *create_test_cases(text_input_test_case_builder),
        *create_test_cases(tools_test_case_builder),
    ],
    ids=lambda tc: tc.get_id(),
)
@pytest.mark.asyncio
async def test_chat_completion(
    test_case: TestCase,
    get_openai_client,
):
    client = get_openai_client(test_case.deployment_config)

    async def run_chat_completion() -> ChatCompletionResult:
        return await chat_completion(
            client,
            test_case.deployment_config.deployment_id,
            test_case.messages,
            test_case.streaming,
            test_case.stop,
            test_case.max_tokens,
            test_case.n,
            test_case.functions,
            test_case.tools,
            test_case.temperature,
        )

    if isinstance(test_case.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run_chat_completion()

        actual_exc = exc_info.value

        assert isinstance(actual_exc, test_case.expected.type)
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test_case.expected.status_code
        assert re.search(test_case.expected.message, str(actual_exc))
    else:
        actual_output = await run_chat_completion()
        assert test_case.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
