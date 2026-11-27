from fastapi import APIRouter, FastAPI

from aidial_adapter_openai.decisions.endpoint import decisions_create


def mount_decisions_api(app: FastAPI, path: str) -> None:
    router = APIRouter(prefix=path)
    router.add_api_route("/v1/decisions", decisions_create, methods=["POST"])
    app.include_router(router)
