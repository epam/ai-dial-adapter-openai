from typing import Self, TypedDict

from vertexai.preview.language_models import ChatModel, CodeChatModel

from llm.chat_adapter import ChatAdapter
from llm.vertex_ai import (
    get_vertex_ai_chat_model,
    get_vertex_ai_code_chat_model,
    init_vertex_ai,
)
from universal_api.request import CompletionParameters
from utils.log_config import vertex_ai_logger as log


class ChatModelParameters(TypedDict, total=False):
    max_output_tokens: int
    temperature: float
    top_p: float
    top_k: int


class CodeChatModelParameters(TypedDict, total=False):
    max_output_tokens: int
    temperature: float


def prepare_chat_model_kwargs(
    model_params: CompletionParameters,
) -> ChatModelParameters:
    # See chat playground: https://console.cloud.google.com/vertex-ai/generative/language/create/chat
    model_kwargs: ChatModelParameters = {}

    if model_params.max_tokens is not None:
        model_kwargs["max_output_tokens"] = model_params.max_tokens

    if model_params.temperature is not None:
        model_kwargs["temperature"] = model_params.temperature

    if model_params.top_p is not None:
        model_kwargs["top_p"] = model_params.top_p

    if model_params.top_k is not None:
        model_kwargs["top_k"] = model_params.top_k

    return model_kwargs


def prepare_code_chat_model_kwargs(
    model_params: CompletionParameters,
) -> CodeChatModelParameters:
    model_kwargs: CodeChatModelParameters = {}

    if model_params.max_tokens is not None:
        model_kwargs["max_output_tokens"] = model_params.max_tokens

    if model_params.temperature is not None:
        model_kwargs["temperature"] = model_params.temperature

    if model_params.top_p is not None:
        log.warning("top_p is not supported for code chat models")

    if model_params.top_k is not None:
        log.warning("top_k is not supported for code chat models")

    return model_kwargs


class VertexAIChatModel(ChatAdapter):
    def __init__(
        self,
        model: ChatModel,
        model_params: CompletionParameters,
        params: ChatModelParameters,
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
    ) -> Self:
        params = prepare_chat_model_kwargs(model_params)

        await init_vertex_ai(project_id, location)

        model = await get_vertex_ai_chat_model(model_id)
        return cls(model, model_params, params)


class VertexAICodeChatModel(ChatAdapter):
    def __init__(
        self,
        model: CodeChatModel,
        model_params: CompletionParameters,
        params: CodeChatModelParameters,
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
    ) -> Self:
        params = prepare_chat_model_kwargs(model_params)

        await init_vertex_ai(project_id, location)

        model = await get_vertex_ai_code_chat_model(model_id)
        return cls(model, model_params, params)
