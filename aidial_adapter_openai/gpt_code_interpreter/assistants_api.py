import json
import time
from pathlib import Path
from typing import List, Literal, Optional

from openai import AsyncOpenAI
from openai.types.beta.threads.message import Message

from aidial_adapter_openai.utils.log_config import logger as log


async def poll_run_till_completion(
    client: AsyncOpenAI,
    thread_id: str,
    run_id: str,
    available_functions: dict,
    max_steps: int = 10,
    wait: int = 3,
) -> None:

    if (client is None and thread_id is None) or run_id is None:
        print("Client, Thread ID and Run ID are required.")
        return
    try:
        cnt = 0
        while cnt < max_steps:
            run = await client.beta.threads.runs.retrieve(
                thread_id=thread_id, run_id=run_id
            )

            log.debug("Poll {}: {}".format(cnt, run.status))

            cnt += 1
            if run.required_action is not None:
                assert run.status == "requires_action"

                tool_responses = []
                if (
                    run.required_action.type == "submit_tool_outputs"
                    and run.required_action.submit_tool_outputs.tool_calls
                    is not None
                ):
                    tool_calls = (
                        run.required_action.submit_tool_outputs.tool_calls
                    )

                    for call in tool_calls:
                        if call.type == "function":
                            if call.function.name not in available_functions:
                                raise ValueError(
                                    "Function requested by the model does not exist"
                                )
                            function_to_call = available_functions[
                                call.function.name
                            ]
                            tool_response = function_to_call(
                                **json.loads(call.function.arguments)
                            )
                            tool_responses.append(
                                {
                                    "tool_call_id": call.id,
                                    "output": tool_response,
                                }
                            )

                run = await client.beta.threads.runs.submit_tool_outputs(
                    thread_id=thread_id,
                    run_id=run.id,
                    tool_outputs=tool_responses,
                )

            if run.status == "failed":
                raise RuntimeError("The run has failed.")

            if run.status == "completed":
                break

            time.sleep(wait)

    except Exception as e:
        log.exception(e)
        raise e


async def create_message(
    client: AsyncOpenAI,
    thread_id: str,
    role: Literal["user", "assistant"],
    content: str,
    file_ids: List[str] = [],
    metadata: dict = {},
    message_id: Optional[str] = None,
) -> Message | None:

    if client is None:
        print("Client parameter is required.")
        return None

    if thread_id is None:
        print("Thread ID is required.")
        return None

    try:
        if message_id is not None:
            return await client.beta.threads.messages.retrieve(
                thread_id=thread_id, message_id=message_id
            )

        if (
            file_ids is not None
            and len(file_ids) > 0
            and metadata is not None
            and len(metadata) > 0
        ):
            return await client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                file_ids=file_ids,
                metadata=metadata,
            )

        if file_ids is not None and len(file_ids) > 0:
            return await client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                file_ids=file_ids,
            )

        if metadata is not None and len(metadata) > 0:
            return await client.beta.threads.messages.create(
                thread_id=thread_id,
                role=role,
                content=content,
                metadata=metadata,
            )

        return await client.beta.threads.messages.create(
            thread_id=thread_id, role=role, content=content
        )

    except Exception as e:
        print(e)
        return None


async def retrieve_and_print_messages(
    client: AsyncOpenAI,
    thread_id: str,
    out_dir: Optional[str] = None,
) -> str | None:

    try:
        messages = await client.beta.threads.messages.list(thread_id=thread_id)

        prev_role = None

        log.debug("\n\nCONVERSATION:")

        last_assistant_response: str | None = None

        for md in reversed(messages.data):
            if prev_role == "assistant" and md.role == "user":
                log.debug("------ \n")

            for mc in md.content:
                txt_val: str | None = None

                if mc.type == "text":
                    txt_val = mc.text.value

                elif mc.type == "image_file":
                    if out_dir is not None:
                        image_data = await client.files.content(
                            mc.image_file.file_id
                        )

                        out_dir_path = Path(out_dir)
                        out_dir_path.mkdir(parents=True, exist_ok=True)

                        image_path = out_dir_path / (
                            mc.image_file.file_id + ".png"
                        )

                        with image_path.open("wb") as f:
                            f.write(image_data.read())

                if txt_val is not None:
                    if prev_role == md.role:
                        log.debug(txt_val)
                    else:
                        display_role = {
                            "user": "User query",
                            "assistant": "Assistant response",
                        }
                        log.debug(
                            "{}:\n{}".format(display_role[md.role], txt_val)
                        )

                    if md.role == "assistant":
                        last_assistant_response = txt_val

            prev_role = md.role

        return last_assistant_response

    except Exception as e:
        log.exception(e)
        raise e
