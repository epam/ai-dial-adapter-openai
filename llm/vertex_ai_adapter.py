from llm.bison_chat import BisonChatAdapter, BisonCodeChatAdapter
from llm.chat_completion_adapter import ChatCompletionAdapter
from llm.dolly2 import Dolly2Adapter
from llm.embeddings_adapter import EmbeddingsAdapter
from llm.gecko_embeddings import (
    GeckoTextClassificationEmbeddingsAdapter,
    GeckoTextClusteringEmbeddingsAdapter,
    GeckoTextGenericEmbeddingsAdapter,
)
from llm.llama2 import Llama2Adapter
from llm.vertex_ai_deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
    ExtraChatCompletionDeployment,
)
from universal_api.request import CompletionParameters


async def get_chat_completion_model(
    deployment: ChatCompletionDeployment | ExtraChatCompletionDeployment,
    project_id: str,
    location: str,
    model_params: CompletionParameters,
) -> ChatCompletionAdapter:
    match deployment:
        case ChatCompletionDeployment.CHAT_BISON_1:
            model_id = deployment.get_model_id()
            return await BisonChatAdapter.create(
                model_id, project_id, location, model_params
            )
        case ChatCompletionDeployment.CODECHAT_BISON_1:
            model_id = deployment.get_model_id()
            return await BisonCodeChatAdapter.create(
                model_id, project_id, location, model_params
            )
        case ExtraChatCompletionDeployment.LLAMA2_7B_CHAT_1:
            return await Llama2Adapter.create(
                project_id, location, model_params
            )
        case ExtraChatCompletionDeployment.DOLLY_V2_7B:
            return await Dolly2Adapter.create(
                project_id, location, model_params
            )


async def get_embeddings_model(
    deployment: EmbeddingsDeployment,
    project_id: str,
    location: str,
) -> EmbeddingsAdapter:
    model_id = deployment.get_model_id()
    match deployment:
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1:
            return await GeckoTextGenericEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1_CLASSIFICATION:
            return await GeckoTextClassificationEmbeddingsAdapter.create(
                model_id, project_id, location
            )
        case EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1_CLUSTERING:
            return await GeckoTextClusteringEmbeddingsAdapter.create(
                model_id, project_id, location
            )
