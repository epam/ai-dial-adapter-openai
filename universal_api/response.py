import json
import time
import uuid
from typing import Generator, List, Literal, Tuple, TypedDict

from fastapi import Response
from pydantic import BaseModel
from starlette.responses import StreamingResponse

from universal_api.token_usage import TokenUsage


class ResponseParameters(BaseModel):
    created: int
    id: str
    model: str
    object: str


def generate_event_stream(
    params: ResponseParameters,
    gen: Generator[dict, None, None],
    usage: TokenUsage,
):
    def event_stream():
        for item in gen:
            if item == {}:
                choice = {"index": 0, "delta": {}, "finish_reason": "stop"}

                # Adding usage to the last chunk.
                # OpenAI itself leaves this field undefined/null, but we provide meaningful usage.
                yield wrap_streaming_chunk(
                    params, {"choices": [choice], "usage": usage.to_dict()}
                )

                yield "data: [DONE]\n\n"
                break

            choice = {"index": 0, "delta": item, "finish_reason": None}
            yield wrap_streaming_chunk(params, {"choices": [choice]})

    return StreamingResponse(
        event_stream(), headers={"Content-Type": "text/event-stream"}
    )


def wrap_streaming_chunk(params: ResponseParameters, payload: dict):
    return (
        "data: "
        + json.dumps(
            {
                **params.dict(),
                **payload,
            }
        )
        + "\n\n"
    )


def wrap_single_message(
    params: ResponseParameters,
    chunk: dict,
    usage: TokenUsage,
) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": chunk,
                "finish_reason": "stop",
            }
        ],
        **params.dict(),
        "usage": usage.to_dict(),
    }


def make_chat_completion_response(
    streaming: bool, model_id: str, name: str, resp: Tuple[str, TokenUsage]
) -> Response | dict:
    id = str(uuid.uuid4())
    timestamp = int(time.time())
    content, usage = resp

    if streaming:
        params = ResponseParameters(
            model=model_id, id=id, created=timestamp, object=name + ".chunk"
        )
        chunks: List[dict] = [{"role": "assistant"}, {"content": content}, {}]
        return generate_event_stream(params, (c for c in chunks), usage)
    else:
        params = ResponseParameters(
            model=model_id, id=id, created=timestamp, object=name
        )
        chunk = {
            "role": "assistant",
            "content": content,
        }
        return wrap_single_message(params, chunk, usage)


class EmbeddingsDict(TypedDict):
    index: int
    object: Literal["embedding"]
    embedding: List[float]


class EmbeddingsTokenUsageDict(TypedDict):
    prompt_tokens: int
    total_tokens: int


class EmbeddingsResponseDict(TypedDict):
    object: Literal["list"]
    model: str
    data: List[EmbeddingsDict]
    usage: EmbeddingsTokenUsageDict


def make_embeddings_response(
    model_id: str, resp: Tuple[List[List[float]], TokenUsage]
) -> EmbeddingsResponseDict:
    vectors, usage = resp

    data: List[EmbeddingsDict] = [
        {"index": idx, "object": "embedding", "embedding": vec}
        for idx, vec in enumerate(vectors)
    ]

    return {
        "object": "list",
        "model": model_id,
        "data": data,
        "usage": usage.to_dict(),
    }
