from enum import Enum


class VllmEmbeddingMode(str, Enum):
    SEQUENCE = "sequence"
    TOKEN_EMBED = "token_embed"


def detect_mode(upstream_endpoint: str) -> VllmEmbeddingMode:
    if upstream_endpoint.rstrip("/").endswith("/pooling"):
        return VllmEmbeddingMode.TOKEN_EMBED
    return VllmEmbeddingMode.SEQUENCE
