from typing import Optional

from aidial_adapter_openai.utils.log_config import logger


def convert_finish_reason(finish_type: Optional[str]) -> Optional[str]:
    match finish_type:
        case None:
            return None
        case "stop":
            return "stop"
        case "max_tokens":
            return "length"
        case "content_filter":
            return "content_filter"
        case _:
            logger.warning(f"Unknown finish type: {finish_type}")
            return None


def convert_gpt4v_to_gpt4_chunk(obj: dict) -> Optional[dict]:
    ret = obj.copy()
    if (
        ret.get("choices", []) == []
        or ret.get("id", "") == ""
        or ret.get("model", "") == ""
        or ret.get("created", 0) == 0
    ):
        return None

    if ret.get("prompt_filter_results"):
        del ret["prompt_filter_results"]

    ret["choices"] = [
        convert_gpt4v_to_gpt4_choice(choice) for choice in obj["choices"]
    ]

    return ret


def convert_gpt4v_to_gpt4_choice(choice: dict) -> dict:
    """GPT4 Vision choice is slightly different from the vanilla GPT4 choice
    in how it reports finish reason."""

    gpt4v_finish_type: Optional[str] = (choice.get("finish_details") or {}).get(
        "type"
    )
    gpt4_finish_reason: Optional[str] = convert_finish_reason(gpt4v_finish_type)

    ret = choice.copy()

    if "finish_details" in ret:
        del ret["finish_details"]

    if "content_filter_results" in ret:
        del ret["content_filter_results"]

    ret["finish_reason"] = gpt4_finish_reason

    return ret
