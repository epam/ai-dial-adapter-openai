from enum import Enum


class EmbeddingAPIType(str, Enum):
    """Upstream vLLM embedding API variants."""

    OPENAI_EMBEDDINGS = "openai_embeddings"
    QWEN3_VL_EMBEDDINGS = "qwen3_vl_embeddings"
    POOLING = "pooling"


def select_api_type(
    model_name: str, upstream_endpoint: str
) -> EmbeddingAPIType:
    if upstream_endpoint.rstrip("/").endswith("/pooling"):
        return EmbeddingAPIType.POOLING

    lower = model_name.lower()
    if "qwen3" in lower and "vl" in lower:
        return EmbeddingAPIType.QWEN3_VL_EMBEDDINGS

    return EmbeddingAPIType.OPENAI_EMBEDDINGS
