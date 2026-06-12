"""
vLLM OpenAI-compatible Embeddings API.

https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#embeddings-api
"""

from dataclasses import dataclass

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from aidial_sdk.exceptions import InvalidRequestError
from pydantic import BaseModel

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource.base import Resource


class VllmOpenAIEmbeddingsRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str
    dimensions: int | None = None
    user: str | None = None


def merge_fanout(model: str, responses: list[dict]) -> EmbeddingResponse:
    vectors: list[Embedding] = []
    prompt_tokens = 0
    total_tokens = 0
    for idx, raw in enumerate(responses):
        parsed = EmbeddingResponse.model_validate(raw)
        item = (
            parsed.data[0]
            if parsed.data
            else Embedding(embedding=[], index=idx)
        )
        vectors.append(Embedding(embedding=item.embedding, index=idx))
        if parsed.usage:
            prompt_tokens += parsed.usage.prompt_tokens
            total_tokens += parsed.usage.total_tokens

    total_tokens = total_tokens or len(vectors)
    prompt_tokens = prompt_tokens or total_tokens

    return EmbeddingResponse(
        model=model,
        data=vectors,
        usage=Usage(
            prompt_tokens=prompt_tokens,
            total_tokens=total_tokens,
        ),
    )


@dataclass
class OpenAIEmbeddingsAdapter:
    request: EmbeddingsRequest
    model: str
    endpoint: str
    creds: OpenAICreds
    headers: dict[str, str] | None

    def _build_request(
        self, input: str | list[str]
    ) -> VllmOpenAIEmbeddingsRequest:
        return VllmOpenAIEmbeddingsRequest(
            model=self.model,
            input=input,
            encoding_format=self.request.encoding_format or "float",
            dimensions=self.request.dimensions,
            user=self.request.user,
        )

    async def _post(self, body: VllmOpenAIEmbeddingsRequest) -> dict:
        return await post_upstream(
            endpoint=self.endpoint,
            body=body.model_dump(exclude_none=True),
            creds=self.creds,
            headers=self.headers,
        )

    async def embeddings(
        self, inputs: list[str | Resource]
    ) -> EmbeddingResponse:
        texts = [item for item in inputs if isinstance(item, str)]
        if len(texts) != len(inputs):
            raise InvalidRequestError(
                "This embedding model supports text inputs only."
            )

        resp = await self._post(self._build_request(texts))
        return EmbeddingResponse.model_validate(resp)
