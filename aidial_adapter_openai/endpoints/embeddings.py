from fastapi import Request

from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.embeddings.azure_ai_vision import (
    embeddings as azure_ai_vision_embeddings,
)
from aidial_adapter_openai.embeddings.openai import (
    embeddings as openai_embeddings,
)
from aidial_adapter_openai.utils.auth import (
    get_credentials_azure,
    get_credentials_vllm,
)
from aidial_adapter_openai.utils.parsers import parse_body
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)


async def embedding(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
    request_body = await parse_body(request)

    # See note for /chat/completions endpoint
    request_body["model"] = request_body.get("model") or deployment_id

    if deployment_id in app_config.VLLM_DEPLOYMENTS:
        creds = get_credentials_vllm(request.headers)
    else:
        creds = await get_credentials_azure(request.headers)

    api_version = get_api_version(request)
    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]
    headers_to_proxy = app_config.get_headers_to_proxy(request.headers)

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
        headers=headers_to_proxy,
    )
