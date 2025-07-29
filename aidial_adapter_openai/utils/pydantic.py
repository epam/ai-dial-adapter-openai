from pydantic import BaseModel


class ExtraAllowedModel(BaseModel):
    class Config:
        extra = "allow"


class ExtraForbidModel(BaseModel):
    class Config:
        extra = "forbid"
