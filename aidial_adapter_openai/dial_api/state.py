from openai.types.responses import ResponseFunctionWebSearch
from pydantic import BaseModel


class MessageState(BaseModel):
    web_search_content: ResponseFunctionWebSearch

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
