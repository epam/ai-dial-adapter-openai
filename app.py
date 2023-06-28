#!/usr/bin/env python3

import logging

import uvicorn
from fastapi import Body, FastAPI, Path
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from llm.vertex_ai import VertexAIModel, vertex_ai_models
from open_ai.response import make_response
from open_ai.types import ChatCompletionQuery, CompletionQuery
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


@app.get("/models")
def models():
    models = [
        ModelDescription(id=model, object="model").dict()
        for model in vertex_ai_models
    ]

    return {"object": "list", "data": models}


@app.post("/{project_id}/chat/completions")
def chat_completions(
    query: ChatCompletionQuery = Body(...),
    project_id: str = Path(...),
):
    model_id = query.model
    model = VertexAIModel(
        model_id=model_id, project_id=project_id, model_params=query
    )
    messages = [message.to_base_message() for message in query.messages]
    response = model.chat(messages)

    streaming = query.stream or False
    return make_response(streaming, model_id, "chat.completion", response)


@app.post("/{project_id}/completions")
def completions(
    query: CompletionQuery = Body(...),
    project_id: str = Path(...),
):
    model_id = query.model
    model = VertexAIModel(
        model_id=model_id, project_id=project_id, model_params=query
    )

    response = model.completion(query.prompt)

    streaming = query.stream or False
    return make_response(streaming, model_id, "text_completion", response)


if __name__ == "__main__":
    init()
    host, port = get_host_port_args()
    uvicorn.run(app, host=host, port=port)
