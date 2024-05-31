from fastapi.responses import StreamingResponse
from openai import AsyncStream
from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

from aidial_adapter_openai.constant import DEFAULT_TIMEOUT
from aidial_adapter_openai.utils.auth import OpenAICreds
from aidial_adapter_openai.utils.exceptions import HTTPException
from aidial_adapter_openai.utils.parsers import chat_completions_parser
from aidial_adapter_openai.utils.reflection import call_with_extra_body
from aidial_adapter_openai.utils.sse_stream import to_openai_sse_stream
from aidial_adapter_openai.utils.streaming import generate_stream, map_stream
from aidial_adapter_openai.utils.tokens import Tokenizer, discard_messages


async def gpt_chat_completion(
    data: dict,
    deployment_id: str,
    upstream_endpoint: str,
    creds: OpenAICreds,
    api_version: str,
    tokenizer: Tokenizer,
):
    discarded_messages = None
    if "max_prompt_tokens" in data:
        max_prompt_tokens = data["max_prompt_tokens"]
        if not isinstance(max_prompt_tokens, int):
            raise HTTPException(
                f"'{max_prompt_tokens}' is not of type 'integer' - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        if max_prompt_tokens < 1:
            raise HTTPException(
                f"'{max_prompt_tokens}' is less than the minimum of 1 - 'max_prompt_tokens'",
                400,
                "invalid_request_error",
            )
        del data["max_prompt_tokens"]

        data["messages"], discarded_messages = discard_messages(
            tokenizer, data["messages"], max_prompt_tokens
        )

    client = chat_completions_parser.parse(upstream_endpoint).get_client(
        {**creds, "api_version": api_version, "timeout": DEFAULT_TIMEOUT}
    )

    response: AsyncStream[ChatCompletionChunk] | ChatCompletion = (
        await call_with_extra_body(client.chat.completions.create, data)
    )

    if isinstance(response, AsyncStream):
        prompt_tokens = tokenizer.calculate_prompt_tokens(data["messages"])
        chunk_stream = map_stream(lambda obj: obj.to_dict(), response)
        return StreamingResponse(
            to_openai_sse_stream(
                generate_stream(
                    prompt_tokens,
                    chunk_stream,
                    tokenizer,
                    deployment_id,
                    discarded_messages,
                )
            ),
            media_type="text/event-stream",
        )
    else:
        resp = response.to_dict()
        if discarded_messages is not None:
            resp |= {"statistics": {"discarded_messages": discarded_messages}}
        return resp
