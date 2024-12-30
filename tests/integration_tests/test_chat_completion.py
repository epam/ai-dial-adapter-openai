import asyncio
import logging
import re
from typing import Generator, List

import pytest
from openai import APIError

from tests.integration_tests.base import TestCase, TestSuite, TestSuiteBuilder
from tests.integration_tests.chat_completion_suites.text import (
    text_common,
    text_multi_system_messages,
    text_stop_sequence,
)
from tests.integration_tests.chat_completion_suites.tools import tools_common
from tests.integration_tests.chat_completion_suites.vision import vision_common
from tests.integration_tests.constants import TEST_DEPLOYMENT_CONFIG
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
        for deployment in TEST_DEPLOYMENT_CONFIG.deployments:
            suite = TestSuite(deployment, streaming)
            for builder in builders:
                builder(suite)
            yield from suite


@pytest.mark.parametrize(
    "test_case",
    create_test_cases(
        [
            text_common,
            text_stop_sequence,
            text_multi_system_messages,
            tools_common,
            vision_common,
        ]
    ),
    ids=lambda tc: tc.get_id(),
)
async def test_chat_completion(
    test_case: TestCase,
    get_openai_client,
):
    client = get_openai_client(test_case.deployment_config)

    async def run_chat_completion() -> ChatCompletionResult:
        for _ in range(3):
            try:
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
            except APIError as e:
                # Somehow, randomly through test, event loop is closing
                if e.message == "Event loop is closed":
                    await asyncio.sleep(5)
                    logger.warning("Event loop is closed, retrying...")
                    continue
                else:
                    raise e
        raise Exception("Event loop retries has failed!")

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
