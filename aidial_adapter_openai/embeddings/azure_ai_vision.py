"""
Adapter for multi-modal embeddings provided by Azure AI Vision service.

1. REST API: https://learn.microsoft.com/en-gb/rest/api/computervision/image-retrieval/vectorize-image?view=rest-computervision-v4.0-preview%20(2023-04-01)&tabs=HTTP
2. How-to article: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/image-retrieval?tabs=python
3. General overview: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/concept-image-retrieval
4. As a plug-in for Azure Search service: https://learn.microsoft.com/en-gb/azure/search/vector-search-vectorizer-ai-services-vision
5. Example of usage in a RAG: https://github.com/Azure-Samples/azure-search-openai-demo/blob/0946893fe904cab1e89de2a38c4421e38d508608/app/backend/prepdocslib/embeddings.py#L226-L260

Note that currently there is no Python SDK for this API.
There is SDK for Image Analysis 4.0 API, but it doesn't cover the multi-modal embeddings API: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-analyze-image-40?pivots=programming-language-python

Input requirements:

1. The file size of the image must be less than 20 megabytes (MB).
2. The dimensions of the image must be greater than 10 x 10 pixels and less than 16,000 x 16,000 pixels.
3. The text string must be between (inclusive) one word and 70 words.

Output characteristics:

1. The vector embeddings are normalized.
2. Image and text vector embeddings have 1024 dimensions.

Limitations:

1. Batching isn't supported.
"""

import asyncio
from typing import AsyncIterator, List, assert_never

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.embedding_inputs import (
    collect_embedding_inputs,
)
from aidial_adapter_openai.dial_api.resource import AttachmentResource
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.http_client import get_http_client

# The latest Image Analysis API offers two models:
# * version 2023-04-15 which supports text search in many languages,
# * the legacy 2022-04-11 model which supports only English.
_VERSION_PARAMS = {
    "api-version": "2024-02-01",
    "model-version": "2023-04-15",
}


def _get_auth_headers(creds: OpenAICreds) -> dict[str, str]:
    if "api_key" in creds:
        return {"Ocp-Apim-Subscription-Key": creds["api_key"]}

    if "azure_ad_token" in creds:
        return {"Authorization": f"Bearer {creds['azure_ad_token']}"}

    raise ValueError("Invalid credentials")


class VectorizeResponse(BaseModel):
    class Config:
        extra = "allow"

    vector: List[float]


async def embeddings(
    creds: OpenAICreds,
    deployment: str,
    endpoint: str,
    file_storage: FileStorage | None,
    data: dict,
) -> EmbeddingResponse:
    input = EmbeddingsRequest.parse_obj(data)

    async def on_text(text: str) -> str | bytes:
        return text

    async def on_attachment(attachment: Attachment) -> str | bytes:
        resource = await AttachmentResource(attachment=attachment).download(
            file_storage
        )
        return resource.data

    inputs: AsyncIterator[str | bytes] = collect_embedding_inputs(
        input,
        on_text=on_text,
        on_attachment=on_attachment,
    )

    async def _get_embedding(input: str | bytes) -> VectorizeResponse:
        if isinstance(input, str):
            return await _get_text_embedding(creds, endpoint, input)
        elif isinstance(input, bytes):
            return await _get_image_embedding(creds, endpoint, input)
        else:
            assert_never(input)

    tasks: List[asyncio.Task[VectorizeResponse]] = []
    async for input in inputs:
        tasks.append(asyncio.create_task(_get_embedding(input)))

    responses = await asyncio.gather(*tasks)
    vectors = [
        Embedding(embedding=r.vector, index=idx)
        for idx, r in enumerate(responses)
    ]

    usage = Usage(prompt_tokens=0, total_tokens=len(vectors))

    return EmbeddingResponse(model=deployment, data=vectors, usage=usage)


async def _get_image_embedding(
    creds: OpenAICreds, endpoint: str, file: bytes
) -> VectorizeResponse:
    resp = await get_http_client().post(
        url=endpoint + ":vectorizeImage",
        # NOTE: when both "text" and "url" fields are provided,
        # the "text" field is ignored.
        files={"file": file},
        headers=_get_auth_headers(creds),
        params=_VERSION_PARAMS,
    )

    resp.raise_for_status()
    return VectorizeResponse.parse_obj(resp.json())


async def _get_text_embedding(
    creds: OpenAICreds, endpoint: str, text: str
) -> VectorizeResponse:
    resp = await get_http_client().post(
        url=endpoint + ":vectorizeText",
        json={"text": text},
        headers=_get_auth_headers(creds),
        params=_VERSION_PARAMS,
    )

    resp.raise_for_status()
    return VectorizeResponse.parse_obj(resp.json())
