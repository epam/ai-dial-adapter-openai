#!/usr/bin/env python3

import logging

from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.vertex_ai import VertexAIModel, vertex_ai_models
from server.exceptions import OpenAIException, error_handling_decorator
from universal_api.request import ChatCompletionQuery, CompletionQuery
from universal_api.response import make_response
from utils.env import get_env
from utils.log_config import LogConfig

logging.config.dictConfig(LogConfig().dict())  # type: ignore

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


class ModelDescription(BaseModel):
    id: str
    object: str


@app.get("/openai/models")
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
@error_handling_decorator
async def chat_completions(
    model_id: str = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: ChatCompletionQuery = Body(...),
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

    streaming = query.stream or False
    return make_response(streaming, model_id, "chat.completion", response)


@app.post("/openai/deployments/{model_id}/completions")
@error_handling_decorator
async def completions(
    model_id: str = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: CompletionQuery = Body(...),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = await VertexAIModel.create(
        location=region,
        model_id=model_id,
        project_id=project_id,
        model_params=query,
    )

    response = await model.completion(query.prompt)

    streaming = query.stream or False
    return make_response(streaming, model_id, "text_completion", response)


@app.exception_handler(OpenAIException)
async def open_ai_exception_handler(request: Request, exc: OpenAIException):
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )
