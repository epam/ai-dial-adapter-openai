from enum import Enum


class VllmEmbeddingMode(str, Enum):
    SEQUENCE = "sequence"
    POOLING = "pooling"


def detect_mode(upstream_endpoint: str) -> VllmEmbeddingMode:
    if upstream_endpoint.rstrip("/").endswith("/pooling"):
        return VllmEmbeddingMode.POOLING
    return VllmEmbeddingMode.SEQUENCE
