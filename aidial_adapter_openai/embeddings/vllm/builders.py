from enum import Enum

from aidial_sdk.exceptions import InvalidRequestError

from aidial_adapter_openai.embeddings.vllm.mode import VllmEmbeddingMode
from aidial_adapter_openai.utils.resource.base import Resource
from aidial_adapter_openai.utils.resource.image import ImageResource

_QWEN3_VL_INSTRUCTION = "Represent the user's input."


class BuilderKind(str, Enum):
    TEXT_INPUT = "text_input"
    QWEN3_VL = "qwen3_vl"
    COLEMBED = "colembed"


def select_builder(model_name: str, mode: VllmEmbeddingMode) -> BuilderKind:
    if mode == VllmEmbeddingMode.TOKEN_EMBED:
        return BuilderKind.COLEMBED

    lower = model_name.lower()
    if "embeddinggemma" in lower:
        return BuilderKind.TEXT_INPUT
    # Qwen3-VL-Embedding is multimodal (messages); Qwen3-Embedding-* is text-only (input).
    if "qwen3" in lower and "vl" in lower:
        return BuilderKind.QWEN3_VL
    return BuilderKind.TEXT_INPUT


def _qwen3_vl_messages(content: list[dict]) -> list[dict]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _QWEN3_VL_INSTRUCTION}],
        },
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": ""}]},
    ]


async def _image_content_part(resource: Resource) -> dict:
    image = await ImageResource.from_resource(resource, detail=None)
    return image.to_content_part()


def _base_request_fields(request: dict, model: str) -> dict:
    body: dict = {
        "model": model,
        "encoding_format": request.get("encoding_format", "float"),
    }
    if (dimensions := request.get("dimensions")) is not None:
        body["dimensions"] = dimensions
    if (user := request.get("user")) is not None:
        body["user"] = user
    return body


async def build_upstream_body(
    *,
    request: dict,
    model: str,
    input_item: str | Resource,
    builder: BuilderKind,
) -> dict:
    if builder == BuilderKind.TEXT_INPUT:
        if isinstance(input_item, Resource):
            raise InvalidRequestError(
                "This embedding model supports text inputs only."
            )
        return {**_base_request_fields(request, model), "input": input_item}

    if builder == BuilderKind.QWEN3_VL:
        body = _base_request_fields(request, model)
        if isinstance(input_item, str):
            content = [{"type": "text", "text": input_item}]
        else:
            image_part = await _image_content_part(input_item)
            content = [image_part, {"type": "text", "text": ""}]
        body["messages"] = _qwen3_vl_messages(content)
        body["continue_final_message"] = True
        body["add_special_tokens"] = True
        return body

    if builder == BuilderKind.COLEMBED:
        body: dict = {"model": model, "task": "token_embed"}
        if isinstance(input_item, str):
            body["input"] = input_item
        else:
            image_part = await _image_content_part(input_item)
            body["messages"] = [
                {
                    "role": "user",
                    "content": [image_part, {"type": "text", "text": ""}],
                }
            ]
        return body

    raise InvalidRequestError(f"Unsupported vLLM embedding builder: {builder}")


async def build_text_batch_body(
    *, request: dict, model: str, texts: list[str]
) -> dict:
    return {**_base_request_fields(request, model), "input": texts}
