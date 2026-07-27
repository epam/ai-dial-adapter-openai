from aidial_sdk.chat_completion.request import ChatCompletionRequest
from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    chat_completions_to_responses_request,
)

_OpenAIClient = AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI


class ResponsesTokenizer:
    _client: _OpenAIClient

    def __init__(
        self, client: _OpenAIClient, file_storage: FileStorage | None
    ) -> None:
        self._client = client
        self._file_storage = file_storage

    async def tokenize_text(self, model_name: str, text: str) -> int:
        response = await self._client.responses.input_tokens.count(
            model=model_name, input=text
        )
        return response.input_tokens

    async def tokenize_request(self, request: ChatCompletionRequest) -> int:
        return await self.tokenize_raw_request(
            request.model_dump(exclude_none=True)
        )

    async def tokenize_raw_request(self, request: dict) -> int:
        tokenize_request, _ = await chat_completions_to_responses_request(
            request=request, file_storage=self._file_storage
        )
        response = await self._client.responses.input_tokens.count(
            **tokenize_request
        )
        return response.input_tokens

    async def tokenize(self, request: dict) -> int:
        return await self.tokenize_raw_request(request)
