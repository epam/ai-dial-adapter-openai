from typing import assert_never

from fastapi import Request

from aidial_adapter_openai.completions import chat_completion as completion
from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType,
)
from aidial_adapter_openai.databricks import (
    chat_completion as databricks_chat_completion,
)
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.gpt import gpt_chat_completion
from aidial_adapter_openai.gpt4_multi_modal.chat_completion import (
    gpt4o_chat_completion,
)
from aidial_adapter_openai.image_generation.generation import (
    chat_completion as image_generation,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.mistral import (
    chat_completion as mistral_chat_completion,
)
from aidial_adapter_openai.responses.adapter import chat_completion as responses
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.image_tokenizer import get_image_tokenizer
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import parse_body
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
    model_name = data["model"] = data.get("model") or deployment_id

    creds = await get_credentials(request)
    api_version = get_api_version(request)

    upstream_endpoint = request.headers.get("X-UPSTREAM-ENDPOINT")
    if upstream_endpoint is None:
        raise ValueError(
            "X-UPSTREAM-ENDPOINT header is missing in the request."
        )

    logger.debug(f"upstream endpoint: {upstream_endpoint}")

    storage = create_file_storage("images", request.headers)

    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )
    logger.debug(f"deployment api type: {deployment.json()}")
    deployment_type, endpoint = deployment.deployment_type, deployment.endpoint

    tiktoken_model = (
        app_config.TIKTOKEN_MODEL_MAPPING.get(deployment_id) or deployment_id
    )

    match deployment_type:
        case ChatCompletionDeploymentType.COMPLETIONS_API:
            return await completion(
                data,
                endpoint,
                creds,
                api_version,
                deployment_id,
                app_config,
            )

        case ChatCompletionDeploymentType.RESPONSES_API:
            return await responses(
                data,
                endpoint,
                creds,
                is_stream,
                storage,
                api_version,
                model_name,
            )

        case (
            ChatCompletionDeploymentType.DALLE3
            | ChatCompletionDeploymentType.GPT_IMAGE_1
        ):
            model = ImageGenerationModel.create(deployment_type)
            return await image_generation(
                model,
                data,
                deployment_id,
                endpoint,
                creds,
                is_stream,
                storage,
                model.get_azure_api_version(app_config),
            )

        case ChatCompletionDeploymentType.MISTRAL:
            return await mistral_chat_completion(data, endpoint, creds)
        case ChatCompletionDeploymentType.DATABRICKS:
            return await databricks_chat_completion(data, endpoint, creds)

        case (
            ChatCompletionDeploymentType.GPT4O
            | ChatCompletionDeploymentType.GPT4O_MINI
        ):
            tokenizer = MultiModalTokenizer(
                tiktoken_model, get_image_tokenizer(deployment_type)
            )
            return await gpt4o_chat_completion(
                data,
                deployment_id,
                request.headers,
                endpoint,
                creds,
                storage,
                api_version,
                tokenizer,
                app_config.ELIMINATE_EMPTY_CHOICES,
            )

        case ChatCompletionDeploymentType.GPT_TEXT_ONLY:
            tokenizer = PlainTextTokenizer(model=tiktoken_model)
            return await gpt_chat_completion(
                data,
                deployment_id,
                endpoint,
                creds,
                api_version,
                tokenizer,
                app_config.ELIMINATE_EMPTY_CHOICES,
            )

        case _:
            assert_never(deployment_type)


async def chat_completion(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
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
