import json
from abc import abstractmethod
from typing import Any, List, Optional, Self, Tuple

from google.cloud.aiplatform import Endpoint
from typing_extensions import override
from vertexai.language_models._language_models import ChatMessage

from llm.chat_completion_adapter import ChatCompletionAdapter
from llm.vertex_ai import init_vertex_ai
from universal_api.request import CompletionParameters
from universal_api.token_usage import TokenUsage
from utils.concurrency import make_async
from utils.log_config import vertex_ai_logger as log


class EndpointChatCompletionAdapter(ChatCompletionAdapter):
    def __init__(
        self,
        model_params: CompletionParameters,
        endpoint: Endpoint,
    ):
        self.model_params = model_params
        self.endpoint = endpoint

    @classmethod
    async def create(
        cls,
        project_id: str,
        location: str,
        endpoint_id: str,
        model_params: CompletionParameters,
    ) -> Self:
        await init_vertex_ai(project_id, location)
        endpoint: Endpoint = Endpoint(
            f"projects/{project_id}/locations/{location}/endpoints/{endpoint_id}"
        )
        return cls(model_params, endpoint)

    @abstractmethod
    async def compile_request(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[List[Any], Optional[str]]:
        pass

    @abstractmethod
    async def parse_response(self, response: Any) -> str:
        pass

    @override
    async def _call(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[str, TokenUsage]:
        instances, opt_prompt = await self.compile_request(
            context, message_history, prompt
        )

        log.debug(f"request:\n{json.dumps(instances, indent=2)}")

        response = await make_async(
            lambda instances: self.endpoint.predict(instances=instances),
            instances,
        )

        log.debug(f"response:\n{json.dumps(response, indent=2)}")

        content = await self.parse_response(response)
        if opt_prompt is not None:
            content = content.removeprefix(opt_prompt)

        return content, TokenUsage.zero_usage()
