from typing import Any, List, Optional, Self, Tuple

from vertexai.language_models._language_models import ChatMessage

from llm.endpoint_chat_completion_adapter import EndpointChatCompletionAdapter
from universal_api.request import CompletionParameters
from utils.env import get_env
from utils.list import list_to_tuples


# Copied from https://huggingface.co/spaces/huggingface-projects/llama-2-13b-chat/blob/main/model.py#L24
def get_llama2_chat_prompt(
    system_prompt: Optional[str],
    chat_history: list[tuple[str, str]],
    message: str,
) -> str:
    texts = (
        [f"<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
        if system_prompt
        else []
    )
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f"{user_input} [/INST] {response.strip()} </s><s>[INST] ")
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


class Llama2Adapter(EndpointChatCompletionAdapter):
    @classmethod
    async def create_endpoint(
        cls, project_id: str, location: str, model_params: CompletionParameters
    ) -> Self:
        return await super().create(
            project_id, location, get_env("LLAMA2_ENDPOINT"), model_params
        )

    async def compile_request(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[List[Any], Optional[str]]:
        messages = list(map(lambda m: m.content, message_history))
        history = list_to_tuples(messages)
        chat_prompt: str = get_llama2_chat_prompt(context, history, prompt)

        instances = [
            {
                "prompt": chat_prompt,
                "max_length": self.model_params.max_tokens or 1024,
                "temperature": self.model_params.temperature or 0.0,
                "n": self.model_params.n or 1,
            },
        ]

        return instances, chat_prompt

    async def parse_response(self, response: Any) -> str:
        return response[0][0][0]["generated_text"]
