from openai.types.create_embedding_response import CreateEmbeddingResponse

from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.parsers import embeddings_parser
from aidial_adapter_openai.utils.reflection import call_with_extra_body


async def embeddings(
    creds: OpenAICreds,
    upstream_endpoint: str,
    api_version: str,
    data: dict,
) -> CreateEmbeddingResponse:

    client = embeddings_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version}
    )

    return await call_with_extra_body(client.embeddings.create, data)
