"""
Qwen3-VL-Embedding via the vLLM OpenAI-compatible Embeddings API.

https://docs.vllm.ai/en/latest/models/pooling_models/embed/
"""

import asyncio

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import EmbeddingResponse

from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.media import image_content_part
from aidial_adapter_openai.embeddings.vllm.openai_api import (
    merge_fanout_embedding_responses,
)
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.pydantic import ExtraAllowedModel
from aidial_adapter_openai.utils.resource.base import Resource

_QWEN3_VL_INSTRUCTION = "Represent the user's input."


class VllmQwen3VLEmbeddingsRequest(ExtraAllowedModel):
    model: str
    messages: list[dict]
    continue_final_message: bool = True
    add_special_tokens: bool = True
    encoding_format: str = "float"
    dimensions: int | None = None
    user: str | None = None


def _qwen3_vl_messages(content: list[dict]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _QWEN3_VL_INSTRUCTION}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


class Qwen3VLEmbeddingsAdapter:
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

    async def build_body(
        self, input_item: str | Resource
    ) -> VllmQwen3VLEmbeddingsRequest:
        if isinstance(input_item, str):
            content = [{"type": "text", "text": input_item}]
        else:
            image_part = await image_content_part(input_item)
            content = [dict(image_part), {"type": "text", "text": ""}]

        return VllmQwen3VLEmbeddingsRequest(
            **self._base_fields(),
            messages=_qwen3_vl_messages(content),
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
        return merge_fanout_embedding_responses(
            model=self._model,
            responses=list(responses),
        )
