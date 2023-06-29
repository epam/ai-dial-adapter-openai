import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import vertexai
from langchain.schema import AIMessage, BaseMessage, SystemMessage
from vertexai.language_models._language_models import ChatMessage
from vertexai.preview.language_models import ChatModel

from llm.chat_model import TokenUsage
from open_ai.types import CompletionParameters
from utils.json import to_json
from utils.text import enforce_stop_tokens
from utils.token_counter import get_num_tokens

vertex_ai_models: List[str] = ["chat-bison@001"]

log = logging.getLogger("vertex-ai")


def compute_usage_estimation(prompt: List[str], completion: str) -> TokenUsage:
    prompt_tokens = get_num_tokens("\n".join(prompt))
    completion_tokens = get_num_tokens(completion)
    token_per_message = 5
    return TokenUsage(
        prompt_tokens=prompt_tokens + len(prompt) * token_per_message,
        completion_tokens=completion_tokens,
    )


def prepare_model_kwargs(model_params: CompletionParameters) -> Dict[str, Any]:
    # See chat playground: https://console.cloud.google.com/vertex-ai/generative/language/create/chat
    model_kwargs = {}

    if model_params.max_tokens is not None:
        model_kwargs["max_output_tokens"] = model_params.max_tokens

    if model_params.temperature is not None:
        model_kwargs["temperature"] = model_params.temperature

    if model_params.top_p is not None:
        model_kwargs["top_p"] = model_params.top_p

    if model_params.top_k is not None:
        model_kwargs["top_k"] = model_params.top_k

    return model_kwargs


def to_chat_message(message: BaseMessage) -> ChatMessage:
    author = "bot" if isinstance(message, AIMessage) else "user"
    return ChatMessage(author=author, content=message.content)


class VertexAIModel:
    def __init__(
        self,
        model_id: str,
        project_id: str,
        location: str,
        model_params: CompletionParameters,
    ):
        self.model_id = model_id
        self.model_params = model_params

        vertexai.init(project=project_id, location=location)

        self.params = prepare_model_kwargs(model_params)
        self.model = ChatModel.from_pretrained(self.model_id)

    def _call(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[str, TokenUsage]:
        prompt_log = json.dumps(
            to_json(
                {
                    "context": context,
                    "messages": message_history,
                    "prompt": prompt,
                }
            ),
            indent=2,
        )
        log.debug(f"prompt:\n{prompt_log}")

        chat = self.model.start_chat(
            context=context,
            message_history=message_history,
            **self.params,
        )

        response = chat.send_message(prompt).text
        log.debug(f"response:\n{response}")

        if self.model_params.stop is not None:
            response = enforce_stop_tokens(response, self.model_params.stop)

        messages: List[str] = []
        if context is not None:
            messages.append(context)
        messages.extend(map(lambda m: m.content, message_history))
        messages.append(prompt)

        usage = compute_usage_estimation(messages, response)

        return response, usage

    def completion(self, prompt: str) -> Tuple[str, TokenUsage]:
        return self._call(None, [], prompt)

    def chat(self, history: List[BaseMessage]) -> Tuple[str, TokenUsage]:
        messages = history.copy()

        context: Optional[str] = None
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            context = messages.pop(0).content

        if len(messages) == 0:
            raise Exception(
                "The chat message must have at least one message besides initial system message"
            )

        message_history = list(map(to_chat_message, messages[:-1]))

        return self._call(context, message_history, messages[-1].content)
