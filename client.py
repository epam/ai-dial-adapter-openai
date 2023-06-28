import json
from typing import List

import requests
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.callback import CallbackWithNewLines
from utils.args import get_host_port_args
from utils.cli import select_option
from utils.env import get_env, get_project_id
from utils.init import init
from utils.input import make_input
from utils.printing import print_ai, print_info


def get_available_models(base_url: str) -> List[str]:
    resp = requests.get(f"{base_url}/models").json()
    models = [r["id"] for r in resp["data"]]
    return models


if __name__ == "__main__":
    init()
    project_id = get_project_id()

    host, port = get_host_port_args()
    base_url = f"http://{host}:{port}"

    model = select_option("Select the model", get_available_models(base_url))

    streaming = select_option("Streaming?", [True, False])
    callbacks = [CallbackWithNewLines()]
    model = ChatOpenAI(
        callbacks=callbacks,
        model=model,
        openai_api_base=f"{base_url}/{project_id}",
        verbose=True,
        streaming=streaming,
        temperature=0,
        request_timeout=6000,
    )  # type: ignore

    history: List[BaseMessage] = []

    input = make_input()

    while True:
        content = input()
        history.append(HumanMessage(content=content))

        llm_result = model.generate([history])

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
