from typing import Any, List

from fastapi.responses import Response
from openai.types.beta.code_interpreter_tool_param import (
    CodeInterpreterToolParam,
)

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.gpt_code_interpreter.assistants_api import (
    create_message,
    poll_run_till_completion,
    retrieve_and_print_messages,
)
from aidial_adapter_openai.gpt_code_interpreter.bing_search import (
    search_bing,
    search_bing_tool,
)
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.parsers import (
    AzureOpenAIEndpoint,
    chat_completions_parser,
)
from aidial_adapter_openai.utils.streaming import create_single_message_response


async def chat_completion(
    request: Any,
    stream: bool,
    endpoint: str,
    creds: OpenAICreds,
    api_version: str,
) -> Response:
    pass

    openai_endpoint = chat_completions_parser.parse(endpoint)

    if not isinstance(openai_endpoint, AzureOpenAIEndpoint):
        raise HTTPException(
            "Invalid upstream endpoint format", 400, "invalid_request_error"
        )

    model = openai_endpoint.azure_deployment

    client = openai_endpoint.get_assistants_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    messages: List[dict] = request["messages"]  # type: ignore

    instructions: str | None = None
    if len(messages) > 0 and messages[0]["role"] == "system":
        instructions = messages[0]["content"]
        messages = messages[1:]

    query = messages[-1]["content"]

    code_interpreter: CodeInterpreterToolParam = {"type": "code_interpreter"}

    assistant = await client.beta.assistants.create(
        model=model,
        instructions=instructions,
        tools=[code_interpreter, search_bing_tool],
    )

    thread = await client.beta.threads.create()
    await create_message(client, thread.id, "user", query)

    run = await client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=instructions,
    )

    await poll_run_till_completion(
        client=client,
        thread_id=thread.id,
        run_id=run.id,
        available_functions={"search_bing": search_bing},
    )

    content = await retrieve_and_print_messages(
        client=client, thread_id=thread.id, out_dir="data"
    )

    return create_single_message_response(content or "NA", stream)
