from typing import Any


def is_request_used_functions_or_tools(data: Any):
    if (
        "functions" in data
        or "function_call" in data
        or "tools" in data
        or "tool_choice" in data
    ):
        return True

    if "messages" not in data:
        return False

    for message in data["messages"]:
        role = message.get("role", None)

        if (
            "tool_calls" in message
            or "function_call" in message
            or role in ["tool", "function"]
        ):
            return True

    return False
