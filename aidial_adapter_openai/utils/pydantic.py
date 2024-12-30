from pydantic import BaseModel


class ExtraAllowedModel(BaseModel):
    class Config:
        extra = "allow"
