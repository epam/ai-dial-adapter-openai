import re
from dataclasses import dataclass
from typing import Callable, Generator, List

import openai
import pytest
from openai.types import CreateEmbeddingResponse

from tests.integration_tests.base import (
    DeploymentConfig,
    EmbeddingsDeploymentType,
    sanitize_id_part,
)
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.openai import ExpectedException


@dataclass
class TestCase:
    __test__ = False

    deployment_config: DeploymentConfig[EmbeddingsDeploymentType]

    name: str
    input: str | List[str]

    expected: Callable[[CreateEmbeddingResponse], bool] | ExpectedException

    extra_body: dict | None = None

    def get_id(self):
        upstream_idx = self.deployment_config.upstream_idx
        parts = [
            sanitize_id_part(self.name),
            sanitize_id_part(self.deployment_config.id_),
            *([] if upstream_idx is None else [f"upstream:{upstream_idx}"]),
        ]

        return "/".join(parts)


def create_test_cases() -> Generator[TestCase, None, None]:
    for deployment in TEST_DEPLOYMENTS_CONFIG.embedding_deployment:

        def check_response(resp: CreateEmbeddingResponse) -> bool:
            assert len(resp.data) == 1
            return True

        yield TestCase(
            name="single text input",
            input="cat",
            deployment_config=deployment,
            expected=check_response,
        )


@pytest.mark.parametrize(
    "test_case",
    create_test_cases(),
    ids=lambda tc: tc.get_id() if isinstance(tc, TestCase) else "na",
)
async def test_embeddings(create_openai_client, test_case: TestCase):
    model_id = test_case.deployment_config.model_name
    client: openai.AsyncAzureOpenAI = create_openai_client(
        test_case.deployment_config
    )

    async def run() -> CreateEmbeddingResponse:
        return await client.embeddings.create(
            model=model_id,
            input=test_case.input,
            extra_body=test_case.extra_body or {},
        )

    if isinstance(test_case.expected, ExpectedException):
        with pytest.raises(Exception) as exc_info:
            await run()

        actual_exc = exc_info.value

        assert isinstance(actual_exc, test_case.expected.type)
        actual_status_code = getattr(actual_exc, "status_code", None)
        assert actual_status_code == test_case.expected.status_code
        assert re.search(test_case.expected.message, str(actual_exc))
    else:
        actual_output = await run()
        assert test_case.expected(
            actual_output
        ), f"Failed output test, actual output: {actual_output}"
