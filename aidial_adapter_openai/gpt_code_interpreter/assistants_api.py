import asyncio
import json
from pathlib import Path
from typing import List, Literal, Optional

from openai import AsyncOpenAI
from openai.types.beta.threads.image_file_content_block import (
    ImageFileContentBlock,
)
from openai.types.beta.threads.message import Message
from openai.types.beta.threads.run import RequiredAction, Run
from openai.types.beta.threads.run_submit_tool_outputs_params import ToolOutput
from openai.types.beta.threads.text_content_block import TextContentBlock

from aidial_adapter_openai.utils.log_config import logger as log


async def submit_tool_outputs(
    client: AsyncOpenAI,
    thread_id: str,
    run_id: str,
    required_action: RequiredAction,
    available_functions: dict,
) -> Run:

    tool_calls = required_action.submit_tool_outputs.tool_calls
    tool_outputs: List[ToolOutput] = []

    for call in tool_calls:
        if call.type == "function":
            if call.function.name not in available_functions:
                raise ValueError(
                    "Function requested by the model does not exist"
                )
            function_to_call = available_functions[call.function.name]
            tool_response = function_to_call(
                **json.loads(call.function.arguments)
            )
            tool_outputs.append(
                {
                    "tool_call_id": call.id,
                    "output": tool_response,
                }
            )

    return await client.beta.threads.runs.submit_tool_outputs(
        thread_id=thread_id,
        run_id=run_id,
        tool_outputs=tool_outputs,
        stream=False,
    )


async def poll_run_till_completion(
    client: AsyncOpenAI,
    thread_id: str,
    run_id: str,
    available_functions: dict,
    max_steps: int = 10,
    wait: int = 3,
) -> None:

    cnt = 0
    while cnt < max_steps:
        run = await client.beta.threads.runs.retrieve(
            thread_id=thread_id, run_id=run_id
        )

        log.debug(f"Run {run.json()}")
        log.debug(f"Poll #{cnt}: {run.status}")

        cnt += 1

        if run.required_action is not None:
            run = await submit_tool_outputs(
                client,
                thread_id,
                run.id,
                run.required_action,
                available_functions,
            )

        elif run.status in ["cancelling", "cancelling", "failed", "expired"]:
            raise RuntimeError(
                f"The run has finished in an unexpected way: {run.status}."
            )

        elif run.status == "completed":
            break

        await asyncio.sleep(wait)


async def retrieve_message(
    client: AsyncOpenAI,
    thread_id: str,
    message_id: str,
):
    return await client.beta.threads.messages.retrieve(
        thread_id=thread_id, message_id=message_id
    )


async def create_message(
    client: AsyncOpenAI,
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    file_ids: List[str] = [],
    metadata: dict = {},
) -> Message:
    return await client.beta.threads.messages.create(
        thread_id=thread_id,
        role=role,
        content=content,
        file_ids=file_ids,
        metadata=metadata,
    )


async def save_image_file(
    client: AsyncOpenAI, content: ImageFileContentBlock, out_dir: str
) -> None:
    image_data = await client.files.content(content.image_file.file_id)

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)

    image_path = out_dir_path / (content.image_file.file_id + ".png")

    with image_path.open("wb") as f:
        f.write(image_data.read())


async def get_response(
    client: AsyncOpenAI,
    thread_id: str,
    run_id: str,
    out_dir: Optional[str] = None,
) -> str:

    messages = await client.beta.threads.messages.list(
        thread_id=thread_id,
        run_id=run_id,
        order="asc",
    )

    log.debug(f"messages: {messages.json()}")

    ret: str = ""

    for md in messages.data:

        # Kind of finish reason
        # incomplete_details = md.incomplete_details

        for mc in md.content:

            message_content: ImageFileContentBlock | TextContentBlock = mc  # type: ignore

            if message_content.type == "text":
                ret += message_content.text.value

            elif message_content.type == "image_file":
                if out_dir is not None:
                    await save_image_file(client, message_content, out_dir)

    return ret
