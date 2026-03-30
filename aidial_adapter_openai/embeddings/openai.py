from openai.types.create_embedding_response import CreateEmbeddingResponse

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import embeddings_parser
from aidial_adapter_openai.utils.reflection import call_with_extra_body


async def embeddings(
    *,
    request: dict,
    creds: OpenAICreds,
    endpoint: str,
    api_version: str,
    headers: dict[str, str] | None = None,
) -> CreateEmbeddingResponse:
    client = embeddings_parser.parse(endpoint).get_client(
        {**creds, "api_version": api_version, "headers": headers or {}}
    )

    return await call_with_extra_body(client.embeddings.create, request)
