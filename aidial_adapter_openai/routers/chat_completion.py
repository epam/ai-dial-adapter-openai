from typing import Annotated

from fastapi import Depends, Request

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.completions import chat_completion as completion
from aidial_adapter_openai.dalle3 import (
    chat_completion as dalle3_chat_completion,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.gpt import gpt_chat_completion
from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    gpt4_vision_chat_completion,
    gpt4o_chat_completion,
)
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.parsers import completions_parser, parse_body
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)
from aidial_adapter_openai.utils.streaming import create_server_response
from aidial_adapter_openai.utils.tokenizer import (
    MultiModalTokenizer,
    PlainTextTokenizer,
)


async def call_chat_completion(
    deployment_id: str,
    data: dict,
    is_stream: bool,
    request: Request,
    app_config: ApplicationConfig,
):

    # Azure OpenAI deployments ignore "model" request field,
    # since the deployment id is already encoded in the endpoint path.
    # This is not the case for non-Azure OpenAI deployments, so
    # they require the "model" field to be set.
    # However, openai==1.33.0 requires the "model" field for **both**
    # Azure and non-Azure deployments.
    # Therefore, we provide the "model" field for all deployments here.
    # The same goes for /embeddings endpoint.
    data["model"] = deployment_id

    creds = await get_credentials(request)
    api_version = get_api_version(request)

    upstream_endpoint = request.headers["X-UPSTREAM-ENDPOINT"]

    if completions_endpoint := completions_parser.parse(upstream_endpoint):
        return await completion(
            data,
            completions_endpoint,
            creds,
            api_version,
            deployment_id,
            app_config,
        )
    if deployment_id in app_config.DALLE3_DEPLOYMENTS:
        storage = create_file_storage("images", request.headers)
        return await dalle3_chat_completion(
            data,
            upstream_endpoint,
            creds,
            is_stream,
            storage,
            app_config.DALLE3_AZURE_API_VERSION,
        )

    if deployment_id in app_config.MISTRAL_DEPLOYMENTS:
        return await mistral_chat_completion(data, upstream_endpoint, creds)

    if deployment_id in app_config.DATABRICKS_DEPLOYMENTS:
        return await databricks_chat_completion(data, upstream_endpoint, creds)

    text_tokenizer_model = app_config.MODEL_ALIASES.get(
        deployment_id, deployment_id
    )

    if deployment_id in app_config.GPT4_VISION_DEPLOYMENTS:
        tokenizer = MultiModalTokenizer(
            "gpt-4", get_image_tokenizer(deployment_id, app_config)
        )
        return await gpt4_vision_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            creds,
            is_stream,
            create_file_storage("images", request.headers),
            api_version,
            tokenizer,
            app_config.ELIMINATE_EMPTY_CHOICES,
        )

    if deployment_id in (
        *app_config.GPT4O_DEPLOYMENTS,
        *app_config.GPT4O_MINI_DEPLOYMENTS,
    ):
        tokenizer = MultiModalTokenizer(
            text_tokenizer_model,
            get_image_tokenizer(deployment_id, app_config),
        )
        return await gpt4o_chat_completion(
            data,
            deployment_id,
            upstream_endpoint,
            creds,
            is_stream,
            create_file_storage("images", request.headers),
            api_version,
            tokenizer,
            app_config.ELIMINATE_EMPTY_CHOICES,
        )

    tokenizer = PlainTextTokenizer(model=text_tokenizer_model)
    return await gpt_chat_completion(
        data,
        deployment_id,
        upstream_endpoint,
        creds,
        api_version,
        tokenizer,
        app_config.ELIMINATE_EMPTY_CHOICES,
    )


async def chat_completion(
    deployment_id: str,
    request: Request,
    app_config: Annotated[ApplicationConfig, Depends(get_request_app_config)],
):

    data = await parse_body(request)

    is_stream = bool(data.get("stream"))

    emulate_streaming = (
        deployment_id in app_config.NON_STREAMING_DEPLOYMENTS and is_stream
    )

    if emulate_streaming:
        data["stream"] = False

    return create_server_response(
        emulate_streaming,
        await call_chat_completion(
            deployment_id, data, is_stream, request, app_config
        ),
    )
