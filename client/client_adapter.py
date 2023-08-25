import json
from typing import List

import openai
from langchain.callbacks.manager import Callbacks
from langchain.chat_models import AzureChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from openai.error import OpenAIError

from llm.callback import CallbackWithNewLines
from utils.args import get_host_port_args
from utils.cli import select_option
from utils.input import make_input
from utils.printing import print_ai, print_error, print_info

DEFAULT_API_VERSION = "2023-03-15-preview"


def get_available_models(base_url: str) -> List[str]:
    resp = openai.Model.list(
        api_type="azure", api_base=base_url, api_version=DEFAULT_API_VERSION
    )
    assert isinstance(resp, dict)
    models = [r["id"] for r in resp.get("data", [])]
    return models


def print_exception(exc: Exception) -> None:
    if isinstance(exc, OpenAIError):
        print_error(json.dumps(exc.json_body, indent=2))
    else:
        print_error(str(exc))


def create_model(
    base_url: str, model_id: str, streaming: bool
) -> BaseChatModel:
    callbacks: Callbacks = [CallbackWithNewLines()]
    return AzureChatOpenAI(
        deployment_name=model_id,
        callbacks=callbacks,
        openai_api_base=base_url,
        openai_api_version=DEFAULT_API_VERSION,
        openai_api_key="dummy_key",
        verbose=True,
        streaming=streaming,
        temperature=0,
        request_timeout=10,
        client=None,
    )


def main():
    host, port = get_host_port_args()
    base_url = f"http://{host}:{port}"

    model_id = select_option("Select the model", get_available_models(base_url))
    streaming = select_option("Streaming?", [False, True])

    model = create_model(base_url, model_id, streaming)

    history: List[BaseMessage] = []

    input = make_input()

    while True:
        content = input()
        if content == ":clear":
            history = []
            continue
        elif content == ":quit":
            break
        elif content == ":reset":
            main()

        history.append(HumanMessage(content=content))

        try:
            llm_result = model.generate([history])
        except Exception as e:
            print_exception(e)
            history.pop()
            continue

        usage = (
            llm_result.llm_output.get("token_usage", {})
            if llm_result.llm_output
            else {}
        )

        response = llm_result.generations[0][-1].text
        if not streaming:
            print_ai(response.strip())

        print_info("Usage:\n" + json.dumps(usage, indent=2))

        message = AIMessage(content=response)
        history.append(message)


if __name__ == "__main__":
    main()
