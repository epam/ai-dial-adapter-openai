"""
vLLM Pooling API.

https://docs.vllm.ai/en/stable/models/pooling_models/
"""

import asyncio

from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.media import image_content_part
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.resource.base import Resource


class VllmPoolingRequest(ExtraAllowedModel):
    model: str
    task: str = "token_embed"
    input: str | None = None
    messages: list[dict] | None = None


class VllmPoolingDataItem(ExtraAllowedModel):
    data: list[list[float]] = []


class VllmPoolingResponse(ExtraAllowedModel):
    data: list[VllmPoolingDataItem] = []


def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
    if not token_vectors:
        return []

    dim = len(token_vectors[0])
    sums = [0.0] * dim
    for vector in token_vectors:
        for idx, value in enumerate(vector):
            sums[idx] += value

    count = len(token_vectors)
    return [value / count for value in sums]


class PoolingEmbeddingsAdapter:
    def __init__(
        self,
        *,
        model: str,
        endpoint: str,
        creds: OpenAICreds,
        headers: dict[str, str] | None,
    ) -> None:
        self._model = model
        self._endpoint = endpoint
        self._creds = creds
        self._headers = headers

    async def build_body(
        self, input_item: str | Resource
    ) -> VllmPoolingRequest:
        if isinstance(input_item, str):
            return VllmPoolingRequest(model=self._model, input=input_item)

        image_part = await image_content_part(input_item)
        return VllmPoolingRequest(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [image_part, {"type": "text", "text": ""}],
                }
            ],
        )

    def _to_embedding(self, raw: dict, *, index: int) -> Embedding:
        parsed = VllmPoolingResponse.model_validate(raw)
        item = parsed.data[0] if parsed.data else VllmPoolingDataItem()
        return Embedding(
            embedding=_mean_pool(item.data),
            index=index,
        )

    async def embeddings(
        self, inputs: list[str | Resource]
    ) -> EmbeddingResponse:
        async def _embed(input_item: str | Resource) -> dict:
            body = await self.build_body(input_item)
            return await post_upstream(
                endpoint=self._endpoint,
                body=body.model_dump(exclude_none=True),
                creds=self._creds,
                headers=self._headers,
            )

        responses = await asyncio.gather(*[_embed(item) for item in inputs])
        vectors = [
            self._to_embedding(raw, index=idx)
            for idx, raw in enumerate(responses)
        ]
        return EmbeddingResponse(
            model=self._model,
            data=vectors,
            usage=Usage(
                prompt_tokens=len(vectors),
                total_tokens=len(vectors),
            ),
        )
