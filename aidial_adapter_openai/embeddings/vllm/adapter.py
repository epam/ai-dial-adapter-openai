from typing import assert_never

from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import EmbeddingResponse

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.embeddings.inputs import download_embedding_inputs
from aidial_adapter_openai.embeddings.vllm.api_type import EmbeddingAPIType
from aidial_adapter_openai.embeddings.vllm.pooling_api import (
    PoolingEmbeddingsAdapter,
)
from aidial_adapter_openai.embeddings.vllm.protocol import VllmEmbeddingsAdapter
from aidial_adapter_openai.embeddings.vllm.qwen3_vl_api import (
    Qwen3VLEmbeddingsAdapter,
)
from aidial_adapter_openai.utils.auth import OpenAICreds


def _create_adapter(
    api_type: EmbeddingAPIType,
    *,
    request: EmbeddingsRequest,
    model: str,
    endpoint: str,
    creds: OpenAICreds,
    headers: dict[str, str] | None,
) -> VllmEmbeddingsAdapter:
    match api_type:
        case EmbeddingAPIType.QWEN3_VL_EMBEDDINGS:
            return Qwen3VLEmbeddingsAdapter(
                request=request,
                model=model,
                endpoint=endpoint,
                creds=creds,
                headers=headers,
            )
        case EmbeddingAPIType.POOLING:
            return PoolingEmbeddingsAdapter(
                model=model,
                endpoint=endpoint,
                creds=creds,
                headers=headers,
            )
        case _:
            assert_never(api_type)


async def embeddings(
    *,
    request: dict,
    creds: OpenAICreds,
    endpoint: str,
    file_storage: FileStorage | None,
    headers: dict[str, str] | None,
    vllm_api_type: EmbeddingAPIType,
    model: str,
) -> EmbeddingResponse:
    body = EmbeddingsRequest.model_validate(request)
    inputs = await download_embedding_inputs(body, file_storage)
    adapter = _create_adapter(
        vllm_api_type,
        request=body,
        model=model,
        endpoint=endpoint,
        creds=creds,
        headers=headers,
    )
    return await adapter.embeddings(inputs)
