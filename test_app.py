from multiprocessing import Process
from typing import Any, Callable, List

import openai
import pytest
import requests
import uvicorn
from fastapi.testclient import TestClient
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel

from app import app
from client.client_adapter import create_model
from llm.vertex_ai_models import VertexAIModelName
from utils.server import ping_server, wait_for_server

client = TestClient(app)

DEFAULT_API_VERSION = "2023-03-15-preview"
HOST = "0.0.0.0"
PORT = 5001

BASE_URL = f"http://{HOST}:{PORT}"

available_models = [
    VertexAIModelName.CHAT_BISON_1,
    VertexAIModelName.CODECHAT_BISON_1,
]


def run_server():
    uvicorn.run(app, host=HOST, port=PORT)


@pytest.fixture(scope="module")
def server():
    already_exists = ping_server(BASE_URL)

    server_process: Process | None = None
    if not already_exists:
        server_process = Process(target=run_server)
        server_process.start()

    assert wait_for_server(BASE_URL), "Server didn't start in time!"

    yield

    if server_process is not None:
        server_process.terminate()
        server_process.join()


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
    expected_models = list(map(lambda e: e.value, available_models))

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


class ModelTestCase(BaseModel):
    model_id: str
    query: str | List[str]
    test: Callable[[str], bool]

    def get_id(self):
        return f"{self.model_id}: {self.query}"

    def get_history(self) -> List[str]:
        return [self.query] if isinstance(self.query, str) else self.query


def get_test_cases_for_model(model_id: str) -> List[ModelTestCase]:
    ret: List[ModelTestCase] = []

    ret.append(
        ModelTestCase(model_id=model_id, query="2+3=?", test=lambda s: "5" in s)
    )

    ret.append(
        ModelTestCase(
            model_id=model_id,
            query='Reply with "Hello"',
            test=lambda s: "hello" in s.lower(),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        test_case
        for model in available_models
        for test_case in get_test_cases_for_model(model.value)
    ],
    ids=lambda test_case: test_case.get_id(),
)
async def test_bedrock_llm_openai(server, test_case: ModelTestCase):
    streaming = False
    model = create_model(BASE_URL, test_case.model_id, streaming)

    await assert_dialog(
        model, test_case.get_history(), test_case.test, streaming
    )
