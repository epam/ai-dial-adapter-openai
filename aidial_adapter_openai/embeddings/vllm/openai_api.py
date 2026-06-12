"""
vLLM OpenAI-compatible Embeddings API.

https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api
"""

import asyncio

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.resource.base import Resource


class VllmOpenAIEmbeddingsRequest(ExtraAllowedModel):
    model: str
    input: str | list[str]
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None


class VllmOpenAIEmbeddingItem(ExtraAllowedModel):
    embedding: list[float] = []
    index: int | None = None


class VllmOpenAIEmbeddingsUsage(ExtraAllowedModel):
    prompt_tokens: int = 0
    total_tokens: int = 0


class VllmOpenAIEmbeddingsResponse(ExtraAllowedModel):
    data: list[VllmOpenAIEmbeddingItem] = []
    usage: VllmOpenAIEmbeddingsUsage | None = None


def to_embedding_response(
    *, model: str, raw: dict, index_offset: int = 0
) -> EmbeddingResponse:
    parsed = VllmOpenAIEmbeddingsResponse.model_validate(raw)
    usage = parsed.usage or VllmOpenAIEmbeddingsUsage()
    vectors = [
        Embedding(
            embedding=item.embedding,
            index=item.index
            if item.index is not None
            else index_offset + idx,
        )
        for idx, item in enumerate(parsed.data)
    ]
    prompt_tokens = usage.prompt_tokens or len(vectors)
    total_tokens = usage.total_tokens or len(vectors)
    return EmbeddingResponse(
        model=model,
        data=vectors,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        ),
    )


def merge_fanout_embedding_responses(
    *, model: str, responses: list[dict]
) -> EmbeddingResponse:
    vectors: list[Embedding] = []
    prompt_tokens = 0
    total_tokens = 0
    for idx, raw in enumerate(responses):
        parsed = VllmOpenAIEmbeddingsResponse.model_validate(raw)
        item = parsed.data[0] if parsed.data else VllmOpenAIEmbeddingItem()
        vectors.append(Embedding(embedding=item.embedding, index=idx))
        if parsed.usage:
            prompt_tokens += parsed.usage.prompt_tokens
            total_tokens += parsed.usage.total_tokens

    if not total_tokens:
        total_tokens = len(vectors)
    if not prompt_tokens:
        prompt_tokens = total_tokens

    return EmbeddingResponse(
        model=model,
        data=vectors,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        ),
    )


class OpenAIEmbeddingsAdapter:
    def __init__(
        self,
        *,
        request: EmbeddingsRequest,
        model: str,
        endpoint: str,
        creds: OpenAICreds,
        headers: dict[str, str] | None,
    ) -> None:
        self._request = request
        self._model = model
        self._endpoint = endpoint
        self._creds = creds
        self._headers = headers

    def _base_fields(self) -> dict:
        fields: dict = {
            "model": self._model,
            "encoding_format": self._request.encoding_format or "float",
        }
        if self._request.dimensions is not None:
            fields["dimensions"] = self._request.dimensions
        if self._request.user is not None:
            fields["user"] = self._request.user
        return fields

    def build_body(self, text: str) -> VllmOpenAIEmbeddingsRequest:
        return VllmOpenAIEmbeddingsRequest(
            **self._base_fields(),
            input=text,
        )

    def build_batch_body(self, texts: list[str]) -> VllmOpenAIEmbeddingsRequest:
        return VllmOpenAIEmbeddingsRequest(
            **self._base_fields(),
            input=texts,
        )

    async def _post(self, body: VllmOpenAIEmbeddingsRequest) -> dict:
        return await post_upstream(
            endpoint=self._endpoint,
            body=body.model_dump(exclude_none=True),
            creds=self._creds,
            headers=self._headers,
        )

    async def embeddings(
        self, inputs: list[str | Resource]
    ) -> EmbeddingResponse:
        texts = [item for item in inputs if isinstance(item, str)]
        if texts and len(texts) == len(inputs) and len(texts) > 1:
            return to_embedding_response(
                model=self._model,
                raw=await self._post(self.build_batch_body(texts)),
            )

        async def _embed(input_item: str | Resource) -> dict:
            if not isinstance(input_item, str):
                raise InvalidRequestError(
                    "This embedding model supports text inputs only."
                )
            return await self._post(self.build_body(input_item))

        responses = await asyncio.gather(*[_embed(item) for item in inputs])
        return merge_fanout_embedding_responses(
            model=self._model,
            responses=list(responses),
        )
