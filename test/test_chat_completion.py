from dataclasses import dataclass
from test.conftest import BASE_URL, DEFAULT_API_VERSION
from typing import Any, Callable, List

import openai
import pytest
import requests
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage

from client.client_adapter import create_chat_model
from llm.vertex_ai_deployments import ChatCompletionDeployment

deployments = [
    ChatCompletionDeployment.CHAT_BISON_1,
    ChatCompletionDeployment.CODECHAT_BISON_1,
]


def models_request_http() -> Any:
    response = requests.get(f"{BASE_URL}/openai/models")
    assert response.status_code == 200
    return response.json()


def models_request_openai() -> Any:
    return openai.Model.list(
        api_type="azure",
        api_base=BASE_URL,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )


def assert_models_subset(models: Any):
    actual_models = [model["id"] for model in models["data"]]
    expected_models = list(map(lambda e: e.value, deployments))

    assert set(expected_models).issubset(
        set(actual_models)
    ), f"Expected models: {expected_models}, Actual models: {actual_models}"


def test_model_list_http(server):
    assert_models_subset(models_request_http())


def test_model_list_openai(server):
    assert_models_subset(models_request_openai())


async def assert_dialog(
    model: BaseChatModel,
    history: List[str],
    output_predicate: Callable[[str], bool],
    streaming: bool,
):
    messages: List[BaseMessage] = []

    SYS_PREFIX = "SYSTEM: "
    AI_PREFIX = "AI: "
    USER_PREFIX = "USER: "

    for s in history:
        msg: BaseMessage = HumanMessage(content=s)
        if s.startswith(SYS_PREFIX):
            msg = SystemMessage(content=s[len(SYS_PREFIX) :])
        elif s.startswith(AI_PREFIX):
            msg = AIMessage(content=s[len(AI_PREFIX) :])
        elif s.startswith(USER_PREFIX):
            msg = HumanMessage(content=s[len(USER_PREFIX) :])

        messages.append(msg)

    llm_result = await model.agenerate([messages])

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage is None) == streaming

    actual_output = llm_result.generations[0][-1].text

    assert output_predicate(
        actual_output
    ), f"Failed output test, actual output: {actual_output}"


@dataclass
class TestCase:
    __test__ = False

    deployment: ChatCompletionDeployment
    query: str | List[str]
    test: Callable[[str], bool]

    def get_id(self):
        return f"{self.deployment.value}: {self.query}"

    def get_history(self) -> List[str]:
        return [self.query] if isinstance(self.query, str) else self.query


def get_test_cases(
    deployment: ChatCompletionDeployment,
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(deployment=deployment, query="2+3=?", test=lambda s: "5" in s)
    )

    ret.append(
        TestCase(
            deployment=deployment,
            query='Reply with "Hello"',
            test=lambda s: "hello" in s.lower(),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [test_case for model in deployments for test_case in get_test_cases(model)],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    streaming = False
    model = create_chat_model(BASE_URL, test.deployment, streaming)

    await assert_dialog(model, test.get_history(), test.test, streaming)
