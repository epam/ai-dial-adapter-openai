from typing import Any


def does_request_use_functions_or_tools(request: Any):
    if (
        "functions" in request
        or "function_call" in request
        or "tools" in request
        or "tool_choice" in request
    ):
        return True

    if "messages" not in request:
        return False

    for message in request["messages"]:
        if (
            "tool_calls" in message
            or "function_call" in message
            or message.get("role") in ["tool", "function"]
        ):
            return True

    return False
