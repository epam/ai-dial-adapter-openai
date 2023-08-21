import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import vertexai
from langchain.schema import AIMessage, BaseMessage, SystemMessage
from vertexai.language_models._language_models import ChatMessage
from vertexai.preview.language_models import ChatModel

from universal_api.request import CompletionParameters
from universal_api.token_usage import TokenUsage
from utils.json import to_json
from utils.log_config import vertex_ai_logger as log
from utils.text import enforce_stop_tokens

vertex_ai_models: List[str] = ["chat-bison@001"]


async def make_async(func, *args):
    with ThreadPoolExecutor() as executor:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(executor, func, *args)


def compute_usage_estimation(prompt: List[str], completion: str) -> TokenUsage:
    # Extremely rough estimation of the number of tokens used by the model.
    # Make sure to upload tokenizer model upfront if you are going to use any.
    symbols_per_token = 3
    prompt_tokens = len(prompt) // symbols_per_token
    completion_tokens = len(completion) // symbols_per_token
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


cached_init = False


# TODO: For now assume that there will be only one project and location.
# We need to fix it otherwise.
async def init_vertex_ai(project_id: str, location: str):
    global cached_init
    if not cached_init:
        await make_async(
            lambda _: vertexai.init(project=project_id, location=location),
            (),
        )
        cached_init = True


cached_models: Dict[str, ChatModel] = {}


async def get_vertex_ai_model(model_id):
    global cached_models
    if model_id not in cached_models:
        cached_models[model_id] = await make_async(
            lambda id: ChatModel.from_pretrained(id), model_id
        )
    return cached_models[model_id]


class VertexAIModel:
    def __init__(
        self,
        model: ChatModel,
        model_params: CompletionParameters,
        params: Dict[str, Any],
    ):
        self.model = model
        self.model_params = model_params
        self.params = params

    @classmethod
    async def create(
        cls,
        model_id: str,
        project_id: str,
        location: str,
        model_params: CompletionParameters,
    ) -> "VertexAIModel":
        model_id = model_id
        model_params = model_params
        params = prepare_model_kwargs(model_params)

        await init_vertex_ai(project_id, location)

        model = await get_vertex_ai_model(model_id)
        return cls(model, model_params, params)

    async def _call(
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

        response = await make_async(
            lambda prompt: chat.send_message(prompt).text, prompt
        )

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

    async def completion(self, prompt: str) -> Tuple[str, TokenUsage]:
        return await self._call(None, [], prompt)

    async def chat(self, history: List[BaseMessage]) -> Tuple[str, TokenUsage]:
        messages = history.copy()

        context: Optional[str] = None
        if len(messages) > 0 and isinstance(messages[0], SystemMessage):
            context = messages.pop(0).content

        if len(messages) == 0:
            raise Exception(
                "The chat message must have at least one message besides initial system message"
            )

        message_history = list(map(to_chat_message, messages[:-1]))

        return await self._call(context, message_history, messages[-1].content)
