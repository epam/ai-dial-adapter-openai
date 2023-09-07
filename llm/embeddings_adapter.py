from abc import ABC, abstractmethod
from typing import List, Optional, Self, Tuple

from llm.vertex_ai import (
    TextEmbeddingModel,
    get_vertex_ai_embeddings_model,
    init_vertex_ai,
)
from universal_api.request import EmbeddingsType
from universal_api.token_usage import TokenUsage


class EmbeddingsAdapter(ABC):
    def __init__(self, model: TextEmbeddingModel):
        self.model = model

    @classmethod
    async def create(
        cls,
        model_id: str,
        project_id: str,
        location: str,
    ) -> Self:
        await init_vertex_ai(project_id, location)
        model = await get_vertex_ai_embeddings_model(model_id)
        return cls(model)

    @abstractmethod
    async def embeddings(
        self,
        input: str | List[str],
        embedding_instruction: Optional[str],
        embedding_type: EmbeddingsType,
    ) -> Tuple[List[List[float]], TokenUsage]:
        pass
