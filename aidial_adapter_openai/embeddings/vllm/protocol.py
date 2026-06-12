from typing import Protocol

from aidial_sdk.embeddings.response import EmbeddingResponse

from aidial_adapter_openai.utils.resource.base import Resource


class VllmEmbeddingsAdapter(Protocol):
    async def embeddings(
        self, inputs: list[str | Resource]
    ) -> EmbeddingResponse: ...
