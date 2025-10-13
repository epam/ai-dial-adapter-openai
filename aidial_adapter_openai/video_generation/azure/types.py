from enum import Enum
from typing import List

from pydantic import BaseModel


class JobStatus(str, Enum):
    PREPROCESSING = "preprocessing"
    QUEUED = "queued"
    RUNNING = "running"
    PROCESSING = "processing"
    CANCELLED = "cancelled"
    SUCCEEDED = "succeeded"
    FAILED = "failed"

    def __str__(self):
        return self.value


class VideoGeneration(BaseModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/aae85aa3e7e4fda95ea2d3abac0ba1d8159db214/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L16081
    """

    id: str


class MediaItemType(str, Enum):
    image = "image"
    video = "video"

    def __str__(self):
        return self.value


class InpaintItem(BaseModel):
    frame_index: int = 0
    type: MediaItemType
    file_name: str


class CreateVideoGenerationRequest(BaseModel):
    """Modelled following the official spec:
    https://github.com/Azure/azure-rest-api-specs/blob/aae85aa3e7e4fda95ea2d3abac0ba1d8159db214/specification/ai/data-plane/OpenAI.v1/azure-v1-preview-generated.yaml#L6730
    """

    model: str
    prompt: str

    width: int
    height: int
    n_seconds: int | None
    n_variants: int | None
    inpaint_items: List[InpaintItem] | None
