"""
Adapter for multi-modal embeddings provided by Azure AI Vision service.

1. Conceptual overview:  https://aka.ms/image-retrieval
2. How-to article: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/image-retrieval?tabs=python
3. REST API (image url, binary image, text): https://learn.microsoft.com/en-gb/rest/api/computervision/image-retrieval?view=rest-computervision-v4.0-preview%20(2023-04-01)
4. A plug-in for Azure Search service: https://learn.microsoft.com/en-gb/azure/search/vector-search-vectorizer-ai-services-vision
5. Example of usage in a RAG: https://github.com/Azure-Samples/azure-search-openai-demo/blob/0946893fe904cab1e89de2a38c4421e38d508608/app/backend/prepdocslib/embeddings.py#L226-L260

Note that currently there is no Python SDK for this API.
There is SDK for Image Analysis 4.0 API, but it doesn't cover the multi-modal embeddings API: https://learn.microsoft.com/en-us/azure/ai-services/computer-vision/how-to/call-analyze-image-40?pivots=programming-language-python

Input requirements:

1. The file size of the image must be less than 20 megabytes (MB).
2. The dimensions of the image must be greater than 10 x 10 pixels and less than 16,000 x 16,000 pixels.
3. The text string must be between (inclusive) one word and 70 words.
4. Supported media types: "application/octet-stream", "image/jpeg", "image/gif", "image/tiff", "image/bmp", "image/png"

Output characteristics:

1. The vector embeddings are normalized.
2. Image and text vector embeddings have 1024 dimensions.

Limitations:

1. Batching isn't supported.

Note that when both "url" and "text" fields are sent in a request,
the "text" field is ignored.
"""

import asyncio
from typing import AsyncIterator, List, assert_never

import aiohttp
from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage
from aidial_sdk.exceptions import HTTPException as DialException
from pydantic import BaseModel

from aidial_adapter_openai.dial_api.embedding_inputs import (
    collect_embedding_inputs,
)
from aidial_adapter_openai.dial_api.resource import AttachmentResource
from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.resource import Resource

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

    async def on_text(text: str) -> str:
        return text

    async def on_attachment(attachment: Attachment) -> Resource:
        return await AttachmentResource(attachment=attachment).download(
            file_storage
        )

    inputs_iter: AsyncIterator[str | Resource] = collect_embedding_inputs(
        input,
        on_text=on_text,
        on_attachment=on_attachment,
    )

    inputs: List[str | Resource] = [input async for input in inputs_iter]

    async def _get_embedding(
        session: aiohttp.ClientSession, input: str | Resource
    ) -> VectorizeResponse:
        if isinstance(input, str):
            return await _get_text_embedding(session, endpoint, input)
        elif isinstance(input, Resource):
            return await _get_image_embedding(session, endpoint, input)
        else:
            assert_never(input)

    async with aiohttp.ClientSession(
        raise_for_status=_error_handler,
        headers=_get_auth_headers(creds),
    ) as session:
        tasks = [
            asyncio.create_task(_get_embedding(session, input_))
            for input_ in inputs
        ]

        responses = await asyncio.gather(*tasks)

    vectors = [
        Embedding(embedding=r.vector, index=idx)
        for idx, r in enumerate(responses)
    ]

    n = len(vectors)
    usage = Usage(prompt_tokens=n, total_tokens=n)

    return EmbeddingResponse(model=deployment, data=vectors, usage=usage)


async def _get_image_embedding(
    session: aiohttp.ClientSession,
    endpoint: str,
    resource: Resource,
) -> VectorizeResponse:
    resp = await session.post(
        url=endpoint.rstrip("/") + "/computervision/retrieval:vectorizeImage",
        params=_VERSION_PARAMS,
        headers={"content-type": resource.type},
        data=resource.data,
    )

    return VectorizeResponse.parse_obj(await resp.json())


async def _get_text_embedding(
    session: aiohttp.ClientSession,
    endpoint: str,
    text: str,
) -> VectorizeResponse:
    resp = await session.post(
        url=endpoint.rstrip("/") + "/computervision/retrieval:vectorizeText",
        params=_VERSION_PARAMS,
        json={"text": text},
    )

    return VectorizeResponse.parse_obj(await resp.json())


async def _error_handler(response: aiohttp.ClientResponse) -> None:
    # The Azure AI Vision service returns error responses in a format similar to the OpenAI error format
    if not response.ok:
        body = await response.json()
        error = body.get("error") or {}

        message = error.get("message") or response.reason or "Unknown Error"
        code = error.get("code")
        type = error.get("type")
        param = error.get("param")

        raise DialException(
            message=message,
            status_code=response.status,
            type=type,
            param=param,
            code=code,
        )
