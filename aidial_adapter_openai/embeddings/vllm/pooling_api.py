"""
vLLM Pooling API.

https://docs.vllm.ai/en/stable/models/pooling_models/
"""

import asyncio
from dataclasses import dataclass

from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from pydantic import BaseModel

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.media import image_content_part
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource.base import Resource


class VllmPoolingRequest(BaseModel):
    model: str
    task: str
    input: str | None
    messages: list[dict] | None


class VllmPoolingDataItem(BaseModel):
    data: list[list[float]]

    def mean_pooled_embedding(self) -> list[float]:
        if not self.data:
            return []

        dim = len(self.data[0])
        sums = [0.0] * dim
        for vector in self.data:
            for idx, value in enumerate(vector):
                sums[idx] += value

        count = len(self.data)
        return [value / count for value in sums]


class VllmPoolingResponse(BaseModel):
    data: list[VllmPoolingDataItem]

    def to_embedding(self, *, index: int) -> Embedding:
        item = self.data[0] if self.data else VllmPoolingDataItem(data=[])
        return Embedding(embedding=item.mean_pooled_embedding(), index=index)

    @classmethod
    def merge_fanout(
        cls, model: str, responses: list[dict]
    ) -> EmbeddingResponse:
        vectors = [
            cls.model_validate(raw).to_embedding(index=idx)
            for idx, raw in enumerate(responses)
        ]
        return EmbeddingResponse(
            model=model,
            data=vectors,
            usage=Usage(
                prompt_tokens=len(vectors),
                total_tokens=len(vectors),
            ),
        )


@dataclass
class PoolingEmbeddingsAdapter:
    model: str
    endpoint: str
    creds: OpenAICreds
    headers: dict[str, str] | None

    async def build_body(
        self, input_item: str | Resource
    ) -> VllmPoolingRequest:
        if isinstance(input_item, str):
            return VllmPoolingRequest(
                model=self.model,
                task="token_embed",
                input=input_item,
                messages=None,
            )

        image_part = await image_content_part(input_item)
        return VllmPoolingRequest(
            model=self.model,
            task="token_embed",
            input=None,
            messages=[
                {
                    "role": "user",
                    "content": [image_part, {"type": "text", "text": ""}],
                }
            ],
        )

    async def embeddings(
        self, inputs: list[str | Resource]
    ) -> EmbeddingResponse:
        async def _embed(input_item: str | Resource) -> dict:
            body = await self.build_body(input_item)
            return await post_upstream(
                endpoint=self.endpoint,
                body=body.model_dump(exclude_none=True),
                creds=self.creds,
                headers=self.headers,
            )

        responses = await asyncio.gather(*[_embed(item) for item in inputs])
        return VllmPoolingResponse.merge_fanout(self.model, responses)
