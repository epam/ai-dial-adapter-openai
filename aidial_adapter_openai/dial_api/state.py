import pydantic
from openai.types.chat import (
    ChatCompletionMessageParam,
)
from openai.types.responses import ResponseOutputItem
from openai.types.responses.response_input_item_param import (
    ResponseInputItemParam,
)
from pydantic import BaseModel

from aidial_adapter_openai.utils.log_config import logger


class MessageState(BaseModel):
    responses_output: list[ResponseOutputItem]

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


def get_message_content_from_state(
    idx: int, message: ChatCompletionMessageParam
) -> list[ResponseInputItemParam] | None:
    if (cc := message.get("custom_content")) and (state := cc.get("state")):
        try:
            state = MessageState.model_validate(state)
            return [block.to_dict() for block in state.responses_output]  # type: ignore
        except pydantic.ValidationError as e:
            logger.error(
                f"Invalid state at the path 'messages[{idx}].custom_content.state': {e}"
            )

    return None
