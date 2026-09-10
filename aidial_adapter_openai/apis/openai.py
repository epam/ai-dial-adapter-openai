from fastapi import APIRouter, FastAPI

import aidial_adapter_openai.endpoints as endpoints

_RESPONSES_ENDPOINTS = [
    ("POST", "", endpoints.responses_create),
    ("GET", "/{responses_id:str}", endpoints.responses_retrieve),
    ("DELETE", "/{responses_id:str}", endpoints.responses_delete),
    ("POST", "/{responses_id:str}/cancel", endpoints.responses_cancel),
    ("POST", "/input_tokens", endpoints.responses_input_tokens),
]

_CHAT_COMPLETIONS_ENDPOINTS = [
    ("POST", "/chat/completions", endpoints.chat_completion),
    ("POST", "/embeddings", endpoints.embedding),
    ("POST", "/tokenize", endpoints.tokenize),
    ("POST", "/truncate_prompt", endpoints.truncate_prompt),
    ("GET", "/configuration", endpoints.configuration),
]


def mount_openai_api(app: FastAPI, path: str):
    router = APIRouter(prefix=path)

    for method, suffix, endpoint in _RESPONSES_ENDPOINTS:
        router.add_api_route(
            "/v1/responses" + suffix, endpoint, methods=[method]
        )

    for method, suffix, endpoint in _CHAT_COMPLETIONS_ENDPOINTS:
        # Azure OpenAI v1 API
        router.add_api_route("/v1" + suffix, endpoint, methods=[method])
        # Azure OpenAI legacy with an explicit deployment id
        router.add_api_route(
            "/deployments/{deployment_id:path}" + suffix,
            endpoint,
            methods=[method],
        )

    app.include_router(router)
