#!/usr/bin/env python3

import asyncio
from typing import List

from langchain.schema import AIMessage, BaseMessage, HumanMessage

from llm.vertex_ai import VertexAIModel
from universal_api.request import CompletionParameters
from utils.env import get_project_id
from utils.init import init
from utils.input import make_input
from utils.printing import print_ai, print_info


async def main():
    init()

    model = await VertexAIModel.create(
        location="us-central1",
        model_id="chat-bison@001",
        project_id=get_project_id(),
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
