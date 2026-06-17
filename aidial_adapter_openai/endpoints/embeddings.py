from fastapi import Request

from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.embeddings.azure_ai_vision import (
    embeddings as azure_ai_vision_embeddings,
)
from aidial_adapter_openai.embeddings.openai import (
    embeddings as openai_embeddings,
)
from aidial_adapter_openai.embeddings.vllm import embeddings as vllm_embeddings
from aidial_adapter_openai.embeddings.vllm.api_type import (
    EmbeddingAPIType,
    select_api_type,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.parsers import parse_body
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)


async def embedding(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
    request_body = await parse_body(request)

    # See note for /chat/completions endpoint
    model = request_body["model"] = request_body.get("model") or deployment_id

    creds = await get_credentials(
        request.headers,
        azure=app_config.is_azure(deployment_id),
    )
    upstream_extra_headers = get_upstream_extra_headers(request.headers)
    api_version = get_api_version(request)
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]
    vllm_api_type = select_api_type(model, upstream_endpoint)

    if (
        deployment_id in app_config.VLLM_DEPLOYMENTS
        and vllm_api_type != EmbeddingAPIType.OPENAI_EMBEDDINGS
    ):
        file_storage = create_file_storage(request.headers)
        return await vllm_embeddings(
            request=request_body,
            creds=creds,
            endpoint=upstream_endpoint,
            file_storage=file_storage,
            headers=upstream_extra_headers,
            vllm_api_type=vllm_api_type,
            model=model,
        )

    if deployment_id in app_config.AZURE_AI_VISION_DEPLOYMENTS:
        file_storage = create_file_storage(request.headers)
        return await azure_ai_vision_embeddings(
            request=request_body,
            creds=creds,
            endpoint=upstream_endpoint,
            file_storage=file_storage,
        )

    return await openai_embeddings(
        request=request_body,
        creds=creds,
        endpoint=upstream_endpoint,
        api_version=api_version,
        headers=upstream_extra_headers,
    )
