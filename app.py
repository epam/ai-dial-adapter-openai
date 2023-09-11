import json
import logging.config
from typing import Optional

from fastapi import Body, FastAPI, Header, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.vertex_ai_adapter import (
    get_chat_completion_model,
    get_embeddings_model,
)
from llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from server.exceptions import OpenAIException, open_ai_exception_decorator
from universal_api.request import (
    ChatCompletionQuery,
    EmbeddingsQuery,
    EmbeddingsType,
)
from universal_api.response import (
    make_chat_completion_response,
    make_embeddings_response,
)
from utils.env import get_env
from utils.log_config import LogConfig
from utils.log_config import app_logger as log

logging.config.dictConfig(LogConfig().dict())

app = FastAPI(
    description="Vertex AI adapter for OpenAI Chat API",
    version="0.0.1",
)

# CORS

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoints


@app.get("/healthcheck")
def healthcheck():
    return Response("OK")


class ModelDescription(BaseModel):
    id: str
    object: str


@app.get("/openai/models")
@open_ai_exception_decorator
async def models():
    models = [
        ModelDescription(id=model.value, object="model").dict()
        for model in ChatCompletionDeployment
    ]

    return {"object": "list", "data": models}


default_region = get_env("DEFAULT_REGION")
default_user_project_id = get_env("ADAPTER_PROJECT_ID")
user_to_palm_mapping = {default_user_project_id: get_env("GCP_PROJECT_ID")}


@app.post("/openai/deployments/{deployment}/chat/completions")
@open_ai_exception_decorator
async def chat_completions(
    deployment: ChatCompletionDeployment = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: ChatCompletionQuery = Body(
        ..., example=ChatCompletionQuery.example()
    ),
):
    log.debug(f"query:\n{json.dumps(query.dict())}")

    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = await get_chat_completion_model(
        deployment=deployment,
        project_id=project_id,
        location=region,
        model_params=query,
    )
    messages = [message.to_base_message() for message in query.messages]

    streaming = bool(query.stream)
    response = await model.chat(streaming, messages)

    return make_chat_completion_response(
        streaming, deployment, "chat.completion", response
    )


@app.post("/openai/deployments/{deployment}/embeddings")
@open_ai_exception_decorator
async def embeddings(
    embeddings_type: EmbeddingsType = Header(
        alias="X-DIAL-Type", default=EmbeddingsType.SYMMETRIC
    ),
    embeddings_instruction: Optional[str] = Header(
        alias="X-DIAL-Instruction", default=None
    ),
    deployment: EmbeddingsDeployment = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: EmbeddingsQuery = Body(..., example=EmbeddingsQuery.example()),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = await get_embeddings_model(
        location=region,
        deployment=deployment,
        project_id=project_id,
    )

    response = await model.embeddings(
        query.input, embeddings_instruction, embeddings_type
    )

    return make_embeddings_response(deployment, response)


@app.exception_handler(OpenAIException)
async def exception_handler(request: Request, exc: OpenAIException):
    log.exception(f"Exception: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )
