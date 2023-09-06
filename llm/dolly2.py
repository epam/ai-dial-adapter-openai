from typing import Any, List, Optional, Self, Tuple

from vertexai.language_models._language_models import ChatMessage

from llm.endpoint_chat_model import EndpointChatModel
from universal_api.request import CompletionParameters
from utils.env import get_env
from utils.list import list_to_tuples


# TODO: follow this format instead: https://huggingface.co/databricks/dolly-v2-7b/blob/main/instruct_pipeline.py
def get_dolly_chat_prompt(
    system_prompt: Optional[str],
    chat_history: list[tuple[str, str]],
    message: str,
) -> str:
    texts = [f"SYSTEM: {system_prompt}\n\n"] if system_prompt else []
    for user_input, response in chat_history:
        texts.append(f"USER: {user_input}\n\n")
        texts.append(f"ASSISTANT: {response}\n\n")
    texts.append(f"USER: {message}\n\n")
    texts.append("ASSISTANT: ")
    return "".join(texts)


class Dolly2Model(EndpointChatModel):
    @classmethod
    async def create(
        cls, project_id: str, location: str, model_params: CompletionParameters
    ) -> Self:
        return await super().create(
            project_id, location, get_env("DOLLY2_ENDPOINT"), model_params
        )

    async def compile_request(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[List[Any], Optional[str]]:
        messages = list(map(lambda m: m.content, message_history))
        history = list_to_tuples(messages)
        chat_prompt: str = get_dolly_chat_prompt(context, history, prompt)

        return [{"text": chat_prompt}], chat_prompt

    async def parse_response(self, response: Any) -> str:
        return response[0][0]["generated_text"]
