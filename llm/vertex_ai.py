from typing import Dict

import vertexai
from vertexai.preview.language_models import (
    ChatModel,
    CodeChatModel,
    TextEmbeddingModel,
)

from utils.concurrency import make_async

_cached_init = False


# TODO: For now assume that there will be only one project and location.
# We need to fix it otherwise.
async def init_vertex_ai(project_id: str, location: str):
    global _cached_init
    if not _cached_init:
        await make_async(
            lambda _: vertexai.init(project=project_id, location=location),
            (),
        )
        _cached_init = True


_cache1: Dict[str, ChatModel] = {}


async def get_vertex_ai_chat_model(model_id: str):
    global _cache1
    if model_id not in _cache1:
        _cache1[model_id] = await make_async(
            ChatModel.from_pretrained, model_id
        )
    return _cache1[model_id]


_cache2: Dict[str, CodeChatModel] = {}


async def get_vertex_ai_code_chat_model(model_id: str):
    global _cache2
    if model_id not in _cache2:
        _cache2[model_id] = await make_async(
            CodeChatModel.from_pretrained, model_id
        )
    return _cache2[model_id]


_cache3: Dict[str, TextEmbeddingModel] = {}


async def get_vertex_ai_embeddings_model(model_id: str):
    global _cache3
    if model_id not in _cache3:
        _cache3[model_id] = await make_async(
            TextEmbeddingModel.from_pretrained, model_id
        )
    return _cache3[model_id]
