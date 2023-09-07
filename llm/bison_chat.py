import json
import logging
from typing import List, Optional, Tuple

from typing_extensions import override
from vertexai.language_models._language_models import ChatMessage

from llm.chat_model_adapter import ChatModelAdapter, CodeChatModelAdapter
from universal_api.token_usage import TokenUsage
from utils.concurrency import make_async
from utils.json import to_json
from utils.log_config import vertex_ai_logger as log
from utils.text import enforce_stop_tokens


# https://cloud.google.com/vertex-ai/docs/generative-ai/pricing
# > Characters are counted by UTF-8 code points and white space is excluded from the count.
def count_characters(s: str) -> int:
    return len("".join(s.split()))


def compute_usage(prompt: List[str], completion: str) -> TokenUsage:
    return TokenUsage(
        prompt_tokens=count_characters("".join(prompt)),
        completion_tokens=count_characters(completion),
    )


class BisonChatAdapter(ChatModelAdapter):
    @override
    async def _call(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[str, TokenUsage]:
        if log.isEnabledFor(logging.DEBUG):
            msg = json.dumps(
                to_json(
                    {
                        "context": context,
                        "messages": message_history,
                        "prompt": prompt,
                    }
                ),
                indent=2,
            )
            log.debug(f"prompt:\n{msg}")

        chat = self.model.start_chat(
            context=context,
            message_history=message_history,
            **self.params,
        )

        response = await make_async(chat.send_message, prompt)

        if log.isEnabledFor(logging.DEBUG):
            msg = json.dumps(to_json(response), indent=2)
            log.debug(f"response:\n{msg}")

        response = response.text

        if self.model_params.stop is not None:
            response = enforce_stop_tokens(response, self.model_params.stop)

        messages: List[str] = []
        if context is not None:
            messages.append(context)
        messages.extend(map(lambda m: m.content, message_history))
        messages.append(prompt)

        usage = compute_usage(messages, response)
        return response, usage


class BisonCodeChatAdapter(CodeChatModelAdapter):
    @override
    async def _call(
        self,
        context: Optional[str],
        message_history: List[ChatMessage],
        prompt: str,
    ) -> Tuple[str, TokenUsage]:
        if context is not None:
            log.warning("System message is ignored for the code chat models")

        if log.isEnabledFor(logging.DEBUG):
            msg = json.dumps(
                to_json({"messages": message_history, "prompt": prompt}),
                indent=2,
            )
            log.debug(f"prompt:\n{msg}")

        chat_session = self.model.start_chat(
            message_history=message_history,
            **self.params,
        )

        response = await make_async(chat_session.send_message, prompt)

        if log.isEnabledFor(logging.DEBUG):
            msg = json.dumps(to_json(response), indent=2)
            log.debug(f"response:\n{msg}")

        response = response.text

        if self.model_params.stop is not None:
            response = enforce_stop_tokens(response, self.model_params.stop)

        messages: List[str] = []
        messages.extend(map(lambda m: m.content, message_history))
        messages.append(prompt)

        usage = compute_usage(messages, response)
        return response, usage
