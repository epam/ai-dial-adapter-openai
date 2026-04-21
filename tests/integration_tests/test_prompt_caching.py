import random
from collections.abc import Callable

import openai
import pytest
from openai.types.chat import (
    ChatCompletionMessageParam,
)

from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from tests.integration_tests.base import DeploymentConfig
from tests.integration_tests.constants import (
    TEST_DEPLOYMENTS_CONFIG,
)
from tests.utils.openai import ai, chat_completion, sys, user

D = DeploymentConfig[ChatCompletionDeploymentType]

_auto_caching_deployments: list[D] = [
    d
    for d in TEST_DEPLOYMENTS_CONFIG.chat_deployments
    if d.model_features.autoCachingSupported
]

if _auto_caching_deployments:

    @pytest.fixture(
        params=_auto_caching_deployments, ids=lambda d: d.display_config()
    )
    def auto_caching_deployment(request) -> D:
        return request.param

else:

    @pytest.fixture
    def auto_caching_deployment(request) -> D:
        pytest.skip("No auto-caching deployments were found")


@pytest.fixture(params=[True, False], ids=["stream", "block"])
def stream(request) -> bool:
    return request.param


def _pseudo_random(seed: int, a: int = 0, b: int = 100) -> int:
    return random.Random(seed).randrange(a, b + 1)  # noqa: S311


def _create_prompt(n: int) -> tuple[str, dict[int, int]]:
    lines = []
    answers = {}
    for idx in range(1, n + 1):
        x = _pseudo_random(2 * idx)
        y = _pseudo_random(2 * idx + 1)
        lines.append(f"[{idx}] {x} + {y} = ?")
        answers[idx] = x + y
    return "\n".join(lines), answers


async def test_auto_caching(
    create_openai_client: Callable[..., openai.AsyncAzureOpenAI],
    auto_caching_deployment: D,
    stream: bool,
) -> None:
    message, answers = _create_prompt(400)

    messages: list[ChatCompletionMessageParam] = [sys(message)]

    indices = [151, 132, 267]
    for i, idx in enumerate(indices):
        query = f"Print the expression [{idx}] and compute it."
        answer = str(answers[idx])

        messages.append(user(query))

        response = await chat_completion(
            create_openai_client(auto_caching_deployment),
            stream=stream,
            deployment_id=auto_caching_deployment.model_name,
            messages=messages,
            max_tokens=512,
        )
        assert answer in response.content

        messages.append(ai(response.content))

        assert response.usage is not None

        # Make sure the prompt size is over the token threshold that triggers the implicit caching:
        # * https://platform.claude.com/docs/en/build-with-claude/prompt-caching#cache-limitations (max threshold is 4096 tokens)
        # https://developers.openai.com/api/docs/guides/prompt-caching#how-it-works (enabled from 1024 tokens)
        assert response.usage.prompt_tokens >= 4_096

        if i:
            assert (details := response.usage.prompt_tokens_details) is not None
            assert (cached := details.cached_tokens) is not None
            assert cached > 0
