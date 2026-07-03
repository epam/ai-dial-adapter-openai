from openai import AsyncAzureOpenAI, AsyncBedrockOpenAI, AsyncOpenAI
from openai.types.responses import ResponseTextConfigParam

from aidial_adapter_openai.responses.converter import (
    convert_messages,
    convert_tool_choice,
    convert_tools,
)
from aidial_adapter_openai.responses.request import convert_response_format
from aidial_adapter_openai.utils.multi_modal_message import MultiModalMessage

_OpenAIClient = AsyncAzureOpenAI | AsyncBedrockOpenAI | AsyncOpenAI


class ResponsesTokenizer:
    _client: _OpenAIClient

    def __init__(self, client: _OpenAIClient) -> None:
        self._client = client

    async def tokenize_text(self, model_name: str, text: str) -> int:
        response = await self._client.responses.input_tokens.count(
            model=model_name, input=text
        )
        return response.input_tokens

    async def tokenize_request(
        self, request: dict, messages: list[MultiModalMessage]
    ) -> int:
        # `request` is always a Chat Completions request by the definition of
        # the `GET /tokenize` endpoint (TokenizeInputRequest.value).
        payload = self._parse_completions(request, messages)

        response = await self._client.responses.input_tokens.count(
            **{
                key: value
                for key, value in payload.items()
                if value is not None
            }
        )
        return response.input_tokens

    def _parse_completions(
        self, request: dict, messages: list[MultiModalMessage]
    ) -> dict:
        payload = self._build_base_payload(request)
        payload["input"] = convert_messages(
            [message.raw_message for message in messages]  # type: ignore
        )
        payload["tools"] = (
            convert_tools(request["tools"])
            if request.get("tools") is not None
            else None
        )
        payload["tool_choice"] = (
            convert_tool_choice(request["tool_choice"])
            if request.get("tool_choice") is not None
            else None
        )
        return payload

    def _build_base_payload(self, request: dict) -> dict:
        payload: dict = {
            "model": request["model"],
            "instructions": request.get("instructions"),
            "conversation": request.get("conversation"),
            "previous_response_id": request.get("previous_response_id"),
            "parallel_tool_calls": request.get("parallel_tool_calls"),
            "reasoning": request.get("reasoning"),
            "truncation": request.get("truncation"),
            "text": request.get("text"),
        }

        if (
            payload["text"] is None
            and request.get("response_format") is not None
        ):
            payload["text"] = ResponseTextConfigParam(
                format=convert_response_format(request["response_format"])
            )
        return payload
