"""
Qwen3-VL-Embedding via vLLM chat-style messages on the /v1/embeddings endpoint.

https://docs.vllm.ai/en/latest/models/pooling_models/embed/
"""

import asyncio
from dataclasses import dataclass

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import EmbeddingResponse
from pydantic import BaseModel

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.media import image_content_part
from aidial_adapter_openai.embeddings.vllm.openai_api import merge_fanout
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource.base import Resource

_QWEN3_VL_DEFAULT_INSTRUCTION = "Represent the user's input."


class VllmQwen3VLEmbeddingsRequest(BaseModel):
    model: str
    messages: list[dict]
    continue_final_message: bool
    add_special_tokens: bool
    encoding_format: str
    dimensions: int | None = None
    user: str | None = None


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
