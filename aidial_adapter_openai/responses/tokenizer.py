from dataclasses import dataclass

from aidial_sdk.chat_completion.request import ChatCompletionRequest
from openai import AsyncOpenAI

from aidial_adapter_openai.dial_api.storage import FileStorage
from aidial_adapter_openai.responses.converter import (
    chat_completions_to_responses_request,
)


@dataclass
class ResponsesRequestTokenizer:
    client: AsyncOpenAI
    file_storage: FileStorage | None

    async def tokenize_text(self, model_name: str, text: str) -> int:
        response = await self.client.responses.input_tokens.count(
            model=model_name, input=text
        )
        return response.input_tokens

    async def tokenize_request(self, request: ChatCompletionRequest) -> int:
        return await self.tokenize_raw_request(
            request.model_dump(exclude_none=True)
        )

    async def tokenize_raw_request(self, request: dict) -> int:
        tokenize_request, _ = await chat_completions_to_responses_request(
            request=request, file_storage=self.file_storage
        )
        response = await self.client.responses.input_tokens.count(
            **tokenize_request
        )
        return response.input_tokens

    async def tokenize(self, request: dict) -> int:
        return await self.tokenize_raw_request(request)
