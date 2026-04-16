from typing import Any

from aidial_sdk.exceptions import RequestValidationError


def validate_request(request: dict[str, Any]) -> None:
    errors: list[str] = []

    if (n := request.get("n")) not in [None, 1]:
        errors.append(
            f"The deployment doesn't support request.n parameter other than 1, but got {n}."
        )

    unsupported_params: list[str] = []
    for param in [
        "stop",
        "seed",
        "top_logprobs",
        "logprobs",
        "presence_penalty",
        "function_call",
        "functions",
        "tools",
        "tool_choice",
    ]:
        if request.get(param) is not None:
            unsupported_params.append(param)

    if unsupported_params:
        suffix = "s" if len(unsupported_params) > 1 else ""
        errors.append(
            f"The deployment doesn't support {', '.join(unsupported_params)} request parameter{suffix}."
        )

    if not (messages := request.get("messages")):
        errors.append("The request doesn't contain any messages.")

    for message in messages or []:
        if message.get("role") in ("system", "developer"):
            errors.append("System and developer messages aren't supported")
            break

    if errors:
        raise RequestValidationError(" ".join(errors))
