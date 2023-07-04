#!/usr/bin/env python3

import logging

import uvicorn
from fastapi import Body, FastAPI, Path, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from llm.vertex_ai import VertexAIModel, vertex_ai_models
from server.exceptions import OpenAIException, error_handling_decorator
from universal_api.request import ChatCompletionQuery, CompletionQuery
from universal_api.response import make_response
from utils.args import get_host_port_args
from utils.init import init
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
def models():
    models = [
        ModelDescription(id=model, object="model").dict()
        for model in vertex_ai_models
    ]

    return {"object": "list", "data": models}


default_region = "us-central1"
default_user_project_id = "EPM-AI-PROXY"
user_to_palm_mapping = {default_user_project_id: "or2-msq-epm-rtc-t1iylu"}


@app.post("/openai/deployments/{model_id}/chat/completions")
@error_handling_decorator
def chat_completions(
    model_id: str = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: ChatCompletionQuery = Body(...),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = VertexAIModel(
        location=region,
        model_id=model_id,
        project_id=project_id,
        model_params=query,
    )
    messages = [message.to_base_message() for message in query.messages]
    response = model.chat(messages)

    streaming = query.stream or False
    return make_response(streaming, model_id, "chat.completion", response)


@app.post("/openai/deployments/{model_id}/completions")
@error_handling_decorator
def completions(
    model_id: str = Path(...),
    project_id: str = Query(
        default=default_user_project_id, description="GCP project"
    ),
    region: str = Query(default=default_region, description="Region"),
    query: CompletionQuery = Body(...),
):
    project_id = user_to_palm_mapping.get(project_id, project_id)
    model = VertexAIModel(
        location=region,
        model_id=model_id,
        project_id=project_id,
        model_params=query,
    )

    response = model.completion(query.prompt)

    streaming = query.stream or False
    return make_response(streaming, model_id, "text_completion", response)


@app.exception_handler(OpenAIException)
async def open_ai_exception_handler(request: Request, exc: OpenAIException):
    return JSONResponse(
        status_code=exc.status_code, content={"error": exc.error}
    )


if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    uvicorn.run(app, host=host, port=port)
