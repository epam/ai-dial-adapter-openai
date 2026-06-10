import asyncio

from aidial_sdk.embeddings.response import (
    Embedding,
    EmbeddingResponse,
    Usage,
)

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.embeddings.inputs import resolve_embedding_inputs
from aidial_adapter_openai.embeddings.vllm.builders import (
    BuilderKind,
    build_text_batch_body,
    build_upstream_body,
    select_builder,
)
from aidial_adapter_openai.embeddings.vllm.client import post_upstream
from aidial_adapter_openai.embeddings.vllm.mode import (
    VllmEmbeddingMode,
    detect_mode,
)
from aidial_adapter_openai.embeddings.vllm.response import to_embedding_response
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource.base import Resource


async def embeddings(
    *,
    request: dict,
    creds: OpenAICreds,
    endpoint: str,
    file_storage: FileStorage | None,
    headers: dict[str, str] | None = None,
) -> EmbeddingResponse:
    inputs = await resolve_embedding_inputs(request, file_storage)
    mode = detect_mode(endpoint)
    model = request["model"]
    builder = select_builder(model, mode)

    texts = [item for item in inputs if isinstance(item, str)]
    if (
        mode == VllmEmbeddingMode.SEQUENCE
        and builder == BuilderKind.TEXT_INPUT
        and texts
        and len(texts) == len(inputs)
        and len(texts) > 1
    ):
        body = await build_text_batch_body(
            request=request, model=model, texts=texts
        )
        response = await post_upstream(
            endpoint=endpoint, body=body, creds=creds, headers=headers
        )
        data = response.get("data") or []
        vectors = [
            Embedding(
                embedding=item.get("embedding") or [],
                index=item.get("index", idx),
            )
            for idx, item in enumerate(data)
        ]
        usage_data = response.get("usage") or {}
        usage = Usage(
            prompt_tokens=int(usage_data.get("prompt_tokens") or len(vectors)),
            total_tokens=int(usage_data.get("total_tokens") or len(vectors)),
        )
        return EmbeddingResponse(model=model, data=vectors, usage=usage)

    async def _embed(input_item: str | Resource) -> dict:
        body = await build_upstream_body(
            request=request,
            model=model,
            input_item=input_item,
            builder=builder,
        )
        return await post_upstream(
            endpoint=endpoint, body=body, creds=creds, headers=headers
        )

    tasks = [asyncio.create_task(_embed(input_item)) for input_item in inputs]
    responses = await asyncio.gather(*tasks)
    return to_embedding_response(model=model, responses=responses, mode=mode)
