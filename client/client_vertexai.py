import asyncio
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.vertex_ai_adapter import get_chat_completion_model
from llm.vertex_ai_deployments import ChatCompletionDeployment
from universal_api.request import CompletionParameters
from utils.cli import select_enum
from utils.env import get_env
from utils.init import init
from utils.input import make_input
from utils.printing import print_ai, print_info


async def main():
    init()

    model_id = select_enum("Select the model", ChatCompletionDeployment)
    model = await get_chat_completion_model(
        location=get_env("DEFAULT_REGION"),
        project_id=get_env("GCP_PROJECT_ID"),
        deployment=model_id,
        model_params=CompletionParameters(),
    )

    history: List[BaseMessage] = []

    input = make_input()

    while True:
        content = input()
        history.append(HumanMessage(content=content))

        response, usage = await model.chat(history)

        print_info(usage.json(indent=2))

        print_ai(response.strip())
        history.append(AIMessage(content=response))


if __name__ == "__main__":
    asyncio.run(main())
