from pydantic import BaseModel, ConfigDict


class ExtraAllowedModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class ExtraForbidModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
