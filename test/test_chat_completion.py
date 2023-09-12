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


def sys(content: str) -> SystemMessage:
    return SystemMessage(content=content)


def ai(content: str) -> AIMessage:
    return AIMessage(content=content)


def user(content: str) -> HumanMessage:
    return HumanMessage(content=content)


async def assert_dialog(
    model: BaseChatModel,
    messages: List[BaseMessage],
    output_predicate: Callable[[str], bool],
    streaming: bool,
):
    llm_result = await model.agenerate([messages])

    actual_usage = (
        llm_result.llm_output.get("token_usage", None)
        if llm_result.llm_output
        else None
    )

    # Usage is missing when and only where streaming is enabled
    assert (actual_usage in [None, {}]) == streaming

    actual_output = llm_result.generations[0][-1].text

    assert output_predicate(
        actual_output
    ), f"Failed output test, actual output: {actual_output}"


@dataclass
class TestCase:
    __test__ = False

    deployment: ChatCompletionDeployment
    streaming: bool

    query: str | List[BaseMessage]
    test: Callable[[str], bool]

    def get_id(self):
        return f"{self.deployment.value}[stream={self.streaming}]: {self.query}"

    def get_messages(self) -> List[BaseMessage]:
        return [user(self.query)] if isinstance(self.query, str) else self.query


def get_test_cases(
    deployment: ChatCompletionDeployment, streaming: bool
) -> List[TestCase]:
    ret: List[TestCase] = []

    ret.append(
        TestCase(
            deployment=deployment,
            streaming=streaming,
            query="2+3=?",
            test=lambda s: "5" in s,
        )
    )

    ret.append(
        TestCase(
            deployment=deployment,
            streaming=streaming,
            query='Reply with "Hello"',
            test=lambda s: "hello" in s.lower(),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test_case
        for model in deployments
        for streaming in [False, True]
        for test_case in get_test_cases(model, streaming)
    ],
    ids=lambda test: test.get_id(),
)
async def test_chat_completion_langchain(server, test: TestCase):
    streaming = test.streaming
    model = create_chat_model(BASE_URL, test.deployment, streaming)
    await assert_dialog(model, test.get_messages(), test.test, streaming)


@dataclass
class ValidationTestCase:
    deployment: ChatCompletionDeployment
    messages: List[BaseMessage]
    expected_error: str

    def get_id(self) -> str:
        return f"{self.expected_error}"


EMPTY_MESSAGE_ERROR = "Empty messages are not allowed"
EMPTY_HISTORY_ERROR = (
    "The chat history must have at least one message besides system message"
)


def get_validation_test_cases(
    deployment: ChatCompletionDeployment,
) -> List[ValidationTestCase]:
    return [
        ValidationTestCase(
            deployment=deployment,
            messages=[],
            expected_error=EMPTY_HISTORY_ERROR,
        ),
        ValidationTestCase(
            deployment=deployment,
            messages=[sys("Act as a helpful assistant")],
            expected_error=EMPTY_HISTORY_ERROR,
        ),
        ValidationTestCase(
            deployment=deployment,
            messages=[user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
        ValidationTestCase(
            deployment=deployment,
            messages=[user("2+2=?"), ai("4"), user("")],
            expected_error=EMPTY_MESSAGE_ERROR,
        ),
    ]


validation_test_cases: List[ValidationTestCase] = [
    test_case
    for deployment in deployments
    for test_case in get_validation_test_cases(deployment)
] + [
    ValidationTestCase(
        deployment=ChatCompletionDeployment.CODECHAT_BISON_1,
        messages=[sys("Act as a helpful assistant"), user("2+2=?")],
        expected_error="System message is not supported",
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test", validation_test_cases, ids=lambda test: test.get_id()
)
async def test_input_validation(server, test: ValidationTestCase):
    streaming = False
    model = create_chat_model(BASE_URL, test.deployment, streaming)

    with pytest.raises(Exception, match=test.expected_error):
        await assert_dialog(model, test.messages, lambda s: True, streaming)
