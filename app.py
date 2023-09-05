import logging.config

from fastapi import Body, FastAPI, Path, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.vertex_ai_adapter import VertexAIModel, vertex_ai_models
from llm.vertex_ai_models import VertexAIModels
from server.exceptions import OpenAIException, open_ai_exception_decorator
from universal_api.request import ChatCompletionQuery, CompletionQuery
from universal_api.response import make_response
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
        ModelDescription(id=model, object="model").dict()
        for model in vertex_ai_models
    ]

    return {"object": "list", "data": models}


default_region = get_env("DEFAULT_REGION")
default_user_project_id = get_env("ADAPTER_PROJECT_ID")
user_to_palm_mapping = {default_user_project_id: get_env("GCP_PROJECT_ID")}


@app.post("/openai/deployments/{model_id}/chat/completions")
@open_ai_exception_decorator
async def chat_completions(
    model_id: VertexAIModels = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: ChatCompletionQuery = Body(
        ..., example=ChatCompletionQuery.example()
    ),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = await VertexAIModel.create(
        location=region,
        model_id=model_id,
        project_id=project_id,
        model_params=query,
    )
    messages = [message.to_base_message() for message in query.messages]
    response = await model.chat(messages)

    return make_response(
        bool(query.stream), model_id, "chat.completion", response
    )


@app.post("/openai/deployments/{model_id}/completions")
@open_ai_exception_decorator
async def completions(
    model_id: VertexAIModels = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: CompletionQuery = Body(..., example=ChatCompletionQuery.example()),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = await VertexAIModel.create(
        location=region,
        model_id=model_id,
        project_id=project_id,
        model_params=query,
    )

    response = await model.completion(query.prompt)

    return make_response(
        bool(query.stream), model_id, "text_completion", response
    )


@app.exception_handler(OpenAIException)
async def exception_handler(request: Request, exc: OpenAIException):
    log.exception(f"Exception: {str(exc)}")
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )
