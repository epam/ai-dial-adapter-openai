from typing import (
    Any,
    AsyncIterator,
    Callable,
    Coroutine,
    List,
    TypeVar,
    assert_never,
    cast,
)

from aidial_sdk.chat_completion.request import Attachment
from aidial_sdk.embeddings.request import EmbeddingsRequest
from aidial_sdk.exceptions import RequestValidationError

_T = TypeVar("_T")

_Coro = Coroutine[_T, Any, Any]
_Tokens = List[int]


async def reject_tokens(tokens: _Tokens):
    raise RequestValidationError(
        "Tokens in an embedding input are not supported. Provide text instead. "
        "When Langchain AzureOpenAIEmbeddings class is used, set 'check_embedding_ctx_length=False' to disable tokenization."
    )


async def reject_mixed(input: List[str | Attachment]):
    raise RequestValidationError(
        "Embedding inputs composed of multiple texts and/or attachments aren't supported"
    )


async def collect_embedding_inputs(
    request: EmbeddingsRequest,
    *,
    on_text: Callable[[str], _Coro[_T]],
    on_attachment: Callable[[Attachment], _Coro[_T]],
    on_tokens: Callable[[_Tokens], _Coro[_T]] = reject_tokens,
    on_mixed: Callable[[List[str | Attachment]], _Coro[_T]] = reject_mixed,
) -> AsyncIterator[_T]:

    async def _on_str_or_attachment(input: str | Attachment) -> _T:
        if isinstance(input, str):
            return await on_text(input)
        elif isinstance(input, Attachment):
            return await on_attachment(input)
        else:
            assert_never(input)

    if isinstance(request.input, str):
        yield await on_text(request.input)
    elif isinstance(request.input, list):

        is_list_of_tokens = False
        for input in request.input:
            if isinstance(input, str):
                yield await on_text(input)
            elif isinstance(input, list):
                yield await on_tokens(input)
            else:
                is_list_of_tokens = True
                break

        if is_list_of_tokens:
            yield await on_tokens(cast(_Tokens, request.input))

    else:
        assert_never(request.input)

    if request.custom_input is None:
        return

    for input in request.custom_input:
        if isinstance(input, (str, Attachment)):
            yield await _on_str_or_attachment(input)
        elif isinstance(input, list):
            if len(input) == 0:
                pass
            elif len(input) == 1:
                yield await _on_str_or_attachment(input[0])
            else:
                yield await on_mixed(input)
        else:
            assert_never(input)
