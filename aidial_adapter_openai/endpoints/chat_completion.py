from collections.abc import Mapping
from typing import assert_never

import fastapi
from anthropic import AsyncAnthropic
from fastapi import Request
from openai import AsyncAzureOpenAI

from aidial_adapter_openai.audio_api.speech.adapter import (
    chat_completion as audio_speech_gen,
)
from aidial_adapter_openai.audio_api.transcribe.adapter import (
    chat_completion as audio_transcriptions_gen,
)
from aidial_adapter_openai.chat_completions.anthropic import (
    chat_completion as anthropic_chat_completions,
)
from aidial_adapter_openai.chat_completions.cache_tokens import (
    normalize_cache_write_tokens,
)
from aidial_adapter_openai.chat_completions.gpt import (
    chat_completion as gpt_chat_completion,
)
from aidial_adapter_openai.chat_completions.gpt_audio import (
    extract_audio_content,
)
from aidial_adapter_openai.chat_completions.gpt_oss import (
    extract_reasoning_tokens,
)
from aidial_adapter_openai.chat_completions.mistral import (
    chat_completion as mistral_gpt_chat_completion,
)
from aidial_adapter_openai.chat_completions.non_gpt import (
    chat_completion as non_gpt_chat_completion,
)
from aidial_adapter_openai.chat_completions.vllm import (
    VllmTokenizer,
)
from aidial_adapter_openai.chat_completions.vllm import (
    chat_completion as vllm_chat_completion,
)
from aidial_adapter_openai.chat_completions.vllm import (
    extract_qwen3_asr_language as vllm_extract_qwen3_asr_language,
)
from aidial_adapter_openai.chat_completions.vllm import (
    extract_reasoning as vllm_extract_reasoning,
)
from aidial_adapter_openai.completions import chat_completion as completion
from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.configuration.deployment_type import (
    ChatCompletionDeploymentType as D,
)
from aidial_adapter_openai.dial_api.request import get_upstream_endpoint
from aidial_adapter_openai.dial_api.storage import create_file_storage
from aidial_adapter_openai.image_generation.adapter import (
    chat_completion as image_generation,
)
from aidial_adapter_openai.image_generation.model import ImageGenerationModel
from aidial_adapter_openai.responses.adapter import chat_completion as responses
from aidial_adapter_openai.utils.auth import get_credentials
from aidial_adapter_openai.utils.log_config import logger
from aidial_adapter_openai.utils.parsers import (
    bad_upstream_endpoint,
    parse_body,
)
from aidial_adapter_openai.utils.request import (
    get_api_version,
    get_request_app_config,
)
from aidial_adapter_openai.utils.streaming import (
    ChatResponse,
    create_server_response,
)
from aidial_adapter_openai.utils.tokenizer import (
    Tokenizer,
    create_tiktoken_tokenizer,
)
from aidial_adapter_openai.utils.upstream_headers import (
    get_upstream_extra_headers,
)
from aidial_adapter_openai.video_generation.azure.adapter import (
    chat_completion as azure_video_gen,
)
from aidial_adapter_openai.video_generation.openai.adapter import (
    chat_completion as openai_video_gen,
)


async def call_chat_completion(
    *,
    app_config: ApplicationConfig,
    deployment_id: str,
    request: fastapi.Request,
    request_body: dict,
    request_headers: Mapping[str, str],
    api_version: str,
) -> ChatResponse:
    # Azure OpenAI deployments ignore "model" request field,
    # since the deployment id is already encoded in the endpoint path.
    # This is not the case for non-Azure OpenAI deployments, so
    # they require the "model" field to be set.
    # However, openai==1.33.0 requires the "model" field for **both**
    # Azure and non-Azure deployments.
    # Therefore, we provide the "model" field for all deployments here.
    # The same goes for /embeddings endpoint.
    model_name = request_body["model"] = (
        request_body.get("model") or deployment_id
    )

    upstream_endpoint = get_upstream_endpoint(request_headers)
    file_storage = create_file_storage(request_headers)

    deployment = app_config.get_chat_completion_deployment_type(
        deployment_id, upstream_endpoint
    )
    logger.debug(f"deployment api type: {deployment.model_dump_json()}")
    deployment_type, endpoint = deployment.deployment_type, deployment.endpoint
    vendor = app_config.get_vendor(deployment_id, endpoint)
    creds = await get_credentials(
        request_headers, vendor=vendor, endpoint=endpoint
    )

    upstream_extra_headers = get_upstream_extra_headers(request_headers)
    client = endpoint.get_client(
        {**creds, "api_version": api_version, "headers": upstream_extra_headers}
    )

    def _get_tokenizer() -> Tokenizer:
        return create_tiktoken_tokenizer(
            app_config, deployment_id, deployment_type
        )

    if isinstance(client, AsyncAnthropic):
        return await anthropic_chat_completions(
            request=request,
            client=client,
            deployment_id=model_name,
        )

    match deployment_type:
        case D.COMPLETIONS_API:
            templates = app_config.COMPLETION_DEPLOYMENTS_PROMPT_TEMPLATES
            prompt_template = templates.get(deployment_id)
            return await completion(
                request=request_body,
                client=client,
                prompt_template=prompt_template,
            )

        case D.RESPONSES_API:
            return await responses(
                request=request_body,
                request_headers=request_headers,
                client=client,
                file_storage=file_storage,
                vendor=vendor,
            )

        case D.AZURE_VIDEO_API:
            return await azure_video_gen(
                request=request,
                request_body=request_body,
                creds=creds,
                deployment_id=deployment_id,
                upstream_endpoint=upstream_endpoint,
                file_storage=file_storage,
            )

        case D.OPENAI_VIDEO_API:
            if isinstance(client, AsyncAzureOpenAI):
                raise bad_upstream_endpoint(
                    "Only v1 API upstream endpoints are supported for OpenAI video generation deployments"
                )

            return await openai_video_gen(
                request=request,
                request_body=request_body,
                client=client,
                deployment_id=deployment_id,
                file_storage=file_storage,
            )

        case D.DALLE3 | D.GPT_IMAGE_1:
            model = ImageGenerationModel.create(deployment_type)

            if isinstance(client, AsyncAzureOpenAI):
                api_version = model.get_azure_api_version(app_config)
                client = client.with_options(api_version=api_version)

            return await image_generation(
                request=request,
                request_body=request_body,
                model=model,
                deployment_id=deployment_id,
                client=client,
                file_storage=file_storage,
            )

        case D.MISTRAL:
            return await mistral_gpt_chat_completion(
                request=request_body, client=client
            )

        case D.DATABRICKS:
            return await non_gpt_chat_completion(
                request=request_body, client=client
            )

        case D.AUDIO_SPEECH_API | D.AUDIO_TRANSCRIPTIONS_API:
            if isinstance(client, AsyncAzureOpenAI):
                client = client.with_options(
                    api_version=app_config.AUDIO_AZURE_API_VERSION
                )

            if deployment_type == D.AUDIO_SPEECH_API:
                return await audio_speech_gen(
                    request=request,
                    request_body=request_body,
                    deployment_id=deployment_id,
                    client=client,
                    file_storage=file_storage,
                    tokenizer=_get_tokenizer(),
                )
            elif deployment_type == D.AUDIO_TRANSCRIPTIONS_API:
                return await audio_transcriptions_gen(
                    request=request,
                    request_body=request_body,
                    deployment_id=deployment_id,
                    client=client,
                    file_storage=file_storage,
                )
            else:
                assert_never(deployment_type)

        case (
            D.VLLM_CHAT_COMPLETIONS_API | D.QWEN3_ASR_VLLM_CHAT_COMPLETIONS_API
        ):
            vllm_tokenizer = VllmTokenizer(
                upstream_endpoint=upstream_endpoint,
                extra_headers=upstream_extra_headers,
            )
            response = await vllm_chat_completion(
                request=request_body,
                client=client,
                file_storage=file_storage,
                tokenizer=vllm_tokenizer,
            )

            if deployment_type == D.VLLM_CHAT_COMPLETIONS_API:
                response.body = vllm_extract_reasoning(response.body)
            else:
                response.body = vllm_extract_qwen3_asr_language(response.body)

            return response

        case D.ANTHROPIC_MESSAGES_API:
            raise RuntimeError(
                "Anthropic API endpoint must have resulted in Anthropic client"
            )

        case D.GPT4O | D.GPT4O_MINI | D.GPT_GENERIC:
            response = await gpt_chat_completion(
                request=request_body,
                request_headers=request_headers,
                client=client,
                file_storage=file_storage,
                tokenizer=_get_tokenizer(),
                eliminate_empty_choices=app_config.ELIMINATE_EMPTY_CHOICES,
            )

            response.body = extract_reasoning_tokens(response.body)
            response.body = extract_audio_content(response.body, request_body)
            response.body = normalize_cache_write_tokens(response.body)

            return response

        case _:
            assert_never(deployment_type)


async def chat_completion(deployment_id: str, request: Request):
    app_config = get_request_app_config(request)
    request_body = await parse_body(request)

    is_stream = bool(request_body.get("stream"))

    emulate_streaming = (
        deployment_id in app_config.NON_STREAMING_DEPLOYMENTS and is_stream
    )

    if emulate_streaming:
        request_body["stream"] = False

    if not is_stream and request_body.get("stream_options") is not None:
        request_body.pop("stream_options")

    api_version = get_api_version(request)

    return await create_server_response(
        await call_chat_completion(
            deployment_id=deployment_id,
            request=request,
            request_body=request_body,
            request_headers=request.headers,
            api_version=api_version,
            app_config=app_config,
        ),
        emulate_streaming=emulate_streaming,
        sse_stream_format="chat-completions",
        sse_heartbeat_interval=app_config.SSE_HEARTBEAT_INTERVAL,
    )
