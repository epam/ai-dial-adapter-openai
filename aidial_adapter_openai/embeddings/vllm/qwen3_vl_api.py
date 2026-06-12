"""
Qwen3-VL-Embedding via vLLM chat-style messages on the /v1/embeddings endpoint.

https://docs.vllm.ai/en/latest/models/pooling_models/embed/
"""

import asyncio
from dataclasses import dataclass

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from pydantic import BaseModel

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.media import image_content_part
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource.base import Resource


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


_QWEN3_VL_DEFAULT_INSTRUCTION = "Represent the user's input."


class VllmQwen3VLEmbeddingsRequest(BaseModel):
    model: str
    messages: list[dict]
    continue_final_message: bool
    add_special_tokens: bool
    encoding_format: str
    dimensions: int | None
    user: str | None


def _instruction(request: EmbeddingsRequest) -> str:
    if request.custom_fields and request.custom_fields.instruction:
        return request.custom_fields.instruction
    return _QWEN3_VL_DEFAULT_INSTRUCTION


def _qwen3_vl_messages(instruction: str, content: list[dict]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": instruction}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


@dataclass
class Qwen3VLEmbeddingsAdapter:
    request: EmbeddingsRequest
    model: str
    endpoint: str
    creds: OpenAICreds
    headers: dict[str, str] | None

    def _build_request(
        self, messages: list[dict]
    ) -> VllmQwen3VLEmbeddingsRequest:
        return VllmQwen3VLEmbeddingsRequest(
            model=self.model,
            messages=messages,
            continue_final_message=True,
            add_special_tokens=True,
            encoding_format=self.request.encoding_format or "float",
            dimensions=self.request.dimensions,
            user=self.request.user,
        )

    async def build_body(
        self, input_item: str | Resource
    ) -> VllmQwen3VLEmbeddingsRequest:
        if isinstance(input_item, str):
            content = [{"type": "text", "text": input_item}]
        else:
            image_part = await image_content_part(input_item)
            content = [dict(image_part), {"type": "text", "text": ""}]

        return self._build_request(
            _qwen3_vl_messages(_instruction(self.request), content)
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
        return merge_fanout(self.model, responses)
