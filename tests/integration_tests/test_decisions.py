"""
Live Decisions API tests.
Every request is billed, so the requests are few and small.
"""

import asyncio
import time
import uuid
from typing import Any

import httpx

from tests.integration_tests.base import (
    DecisionsDeploymentType,
    DeploymentConfig,
)
from tests.integration_tests.constants import TEST_DEPLOYMENTS_CONFIG
from tests.utils.fixtures import maybe_parametrized_fixture

D = DeploymentConfig[DecisionsDeploymentType]

_PATH = "/openai/v1/decisions"

_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"
_CACHE_TTL_SEC = 30 * 60

# The upstream may take a moment to make a fresh cache entry visible
_CACHE_RETRIES = 3
_CACHE_RETRY_DELAY_SEC = 2

_PREDICATE = {
    "type": "predicate",
    "name": "damaged",
    "instructions": "Does the customer report a damaged item?",
}

_QUESTIONS = [
    _PREDICATE,
    {
        "type": "choice",
        "name": "dept",
        "instructions": "Which department should handle the ticket?",
        "choices": [{"value": "billing"}, {"value": "shipping"}],
    },
    {
        "type": "score",
        "name": "sev",
        "instructions": "How severe is the issue?",
        "levels": [{"label": "low"}, {"label": "high"}],
    },
]


@maybe_parametrized_fixture(
    params=list(TEST_DEPLOYMENTS_CONFIG.decisions_deployments),
    ids=lambda d: d.display_config(),
    skip_reason="No Decisions API deployments were found",
)
def decisions_deployment(deployment: D) -> D:
    return deployment


async def _decide(
    test_app: httpx.AsyncClient, deployment: D, body: dict[str, Any]
) -> httpx.Response:
    return await test_app.post(
        _PATH,
        json={"model": deployment.model_name, **body},
        headers={**deployment.upstream_headers, _CACHE_BREAKPOINT_PATH: "x"},
    )


def _usage(response: httpx.Response) -> dict[str, Any]:
    assert response.status_code == 200, response.text
    return response.json()["usage"]


def _cached_tokens(response: httpx.Response) -> int:
    return _usage(response)["input_tokens_details"]["cached_tokens"]


def _assert_cache_headers(response: httpx.Response) -> None:
    now = time.time()
    assert response.headers.get(_CACHE_BREAKPOINT_PATH) == (
        "prefix.body.input[0]"
    )
    expire_at = int(response.headers[_CACHE_EXPIRE_AT])
    assert now + _CACHE_TTL_SEC - 10 <= expire_at <= now + _CACHE_TTL_SEC + 10


def _create_long_input() -> str:
    # The salt keeps a cache warmed by an earlier run from serving
    # the first call
    lines = [f"Ticket log {uuid.uuid4()}."]
    lines += [
        f"[{idx}] Order {idx}: the parcel arrived on day {idx % 28 + 1}"
        f" and the customer rated the delivery {idx % 5 + 1} of 5."
        for idx in range(1, 151)
    ]
    return "\n".join(lines)


async def test_answers_without_cache_headers(
    test_app: httpx.AsyncClient, decisions_deployment: D
) -> None:
    response = await _decide(
        test_app,
        decisions_deployment,
        {
            "input": "The box arrived crushed and the screen is cracked.",
            "questions": _QUESTIONS,
        },
    )

    assert 0 < _usage(response)["input_tokens"] < 1024
    answers = response.json()["answers"]
    assert len(answers) == 3
    assert {answer["type"] for answer in answers} <= {
        "predicate",
        "choice",
        "score",
        "refusal",
    }
    assert _CACHE_BREAKPOINT_PATH not in response.headers
    assert _CACHE_EXPIRE_AT not in response.headers


async def test_prompt_caching(
    test_app: httpx.AsyncClient, decisions_deployment: D
) -> None:
    body = {"input": _create_long_input(), "questions": [_PREDICATE]}

    first = await _decide(test_app, decisions_deployment, body)
    assert _usage(first)["input_tokens"] >= 2000
    _assert_cache_headers(first)

    for attempt in range(_CACHE_RETRIES + 1):
        if attempt:
            await asyncio.sleep(_CACHE_RETRY_DELAY_SEC)
        second = await _decide(test_app, decisions_deployment, body)
        _assert_cache_headers(second)
        if _cached_tokens(second) > 0:
            break

    usage = _usage(second)
    print(
        f"attempts={attempt + 1} input_tokens={usage['input_tokens']}"
        f" cached_tokens={_cached_tokens(second)}"
    )
    assert _cached_tokens(second) > 0


async def test_stream_is_rejected(
    test_app: httpx.AsyncClient, decisions_deployment: D
) -> None:
    response = await _decide(
        test_app,
        decisions_deployment,
        {
            "input": "The box arrived.",
            "questions": [_PREDICATE],
            "stream": True,
        },
    )

    assert response.status_code == 400, response.text
    assert isinstance(response.json()["error"], dict)
