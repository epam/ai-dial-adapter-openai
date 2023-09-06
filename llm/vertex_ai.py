from typing import Dict

import vertexai
from vertexai.preview.language_models import ChatModel, CodeChatModel

from utils.concurrency import make_async

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


_cached_chat_models: Dict[str, ChatModel] = {}


async def get_vertex_ai_chat_model(model_id: str):
    global _cached_chat_models
    if model_id not in _cached_chat_models:
        _cached_chat_models[model_id] = await make_async(
            ChatModel.from_pretrained, model_id
        )
    return _cached_chat_models[model_id]


_cached_code_chat_models: Dict[str, CodeChatModel] = {}


async def get_vertex_ai_code_chat_model(model_id: str):
    global _cached_code_chat_models
    if model_id not in _cached_code_chat_models:
        _cached_code_chat_models[model_id] = await make_async(
            CodeChatModel.from_pretrained, model_id
        )
    return _cached_code_chat_models[model_id]
