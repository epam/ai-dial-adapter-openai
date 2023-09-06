from llm.bison_chat import BisonChatModel, BisonCodeChatModel
from llm.chat_adapter import ChatAdapter
from llm.dolly2 import Dolly2Model
from llm.llama2 import Llama2Model
from llm.vertex_ai_models import ExtraVertexAIModelName, VertexAIModelName
from universal_api.request import CompletionParameters


async def get_model(
    model_id: VertexAIModelName | ExtraVertexAIModelName,
    project_id: str,
    location: str,
    model_params: CompletionParameters,
) -> ChatAdapter:
    match model_id:
        case VertexAIModelName.CHAT_BISON_1:
            return await BisonChatModel.create(
                model_id, project_id, location, model_params
            )
        case VertexAIModelName.CODECHAT_BISON_1:
            return await BisonCodeChatModel.create(
                model_id, project_id, location, model_params
            )
        case ExtraVertexAIModelName.LLAMA2_7B_CHAT_1:
            return await Llama2Model.create(project_id, location, model_params)
        case ExtraVertexAIModelName.DOLLY_V2_7B:
            return await Dolly2Model.create(project_id, location, model_params)
