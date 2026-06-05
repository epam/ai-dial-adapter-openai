from aidial_sdk.embeddings.response import Embedding, EmbeddingResponse, Usage

from aidial_adapter_openai.embeddings.vllm.mode import VllmEmbeddingMode


def _mean_pool(token_vectors: list[list[float]]) -> list[float]:
    if not token_vectors:
        return []

    dim = len(token_vectors[0])
    sums = [0.0] * dim
    for vector in token_vectors:
        for idx, value in enumerate(vector):
            sums[idx] += value

    count = len(token_vectors)
    return [value / count for value in sums]


def _extract_sequence_embedding(response: dict, *, index: int) -> Embedding:
    data = response.get("data") or []
    if not data:
        return Embedding(embedding=[], index=index)

    item = data[0] if len(data) == 1 else data[index]
    return Embedding(embedding=item.get("embedding") or [], index=index)


def _extract_token_embed_embedding(response: dict, *, index: int) -> Embedding:
    data = response.get("data") or []
    if not data:
        return Embedding(embedding=[], index=index)

    item = data[0] if len(data) == 1 else data[index]
    token_vectors = item.get("data") or []
    return Embedding(embedding=_mean_pool(token_vectors), index=index)


def _extract_usage(responses: list[dict], mode: VllmEmbeddingMode) -> Usage:
    prompt_tokens = 0
    total_tokens = 0
    for response in responses:
        usage = response.get("usage") or {}
        prompt_tokens += int(usage.get("prompt_tokens") or 0)
        total_tokens += int(usage.get("total_tokens") or 0)

    if total_tokens == 0 and mode == VllmEmbeddingMode.TOKEN_EMBED:
        total_tokens = len(responses)
    if prompt_tokens == 0 and total_tokens:
        prompt_tokens = total_tokens

    return Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens)


def to_embedding_response(
    *,
    model: str,
    responses: list[dict],
    mode: VllmEmbeddingMode,
) -> EmbeddingResponse:
    extract = (
        _extract_token_embed_embedding
        if mode == VllmEmbeddingMode.TOKEN_EMBED
        else _extract_sequence_embedding
    )
    vectors = [extract(response, index=idx) for idx, response in enumerate(responses)]
    return EmbeddingResponse(
        model=model,
        data=vectors,
        usage=_extract_usage(responses, mode),
    )
