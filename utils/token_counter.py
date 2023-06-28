"""Generic token counter for models which do not provide a token counter."""
from langchain.chat_models import ChatOpenAI


def get_num_tokens(text: str) -> int:
    model = ChatOpenAI(model="gpt-4")  # type: ignore
    return model.get_num_tokens(text)
