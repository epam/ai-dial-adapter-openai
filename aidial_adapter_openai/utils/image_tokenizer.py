"""
Tokenization of images as specified at
    https://learn.microsoft.com/en-us/azure/ai-services/openai/overview#image-tokens
"""

import math
from typing import List, Tuple, assert_never

from pydantic import BaseModel

from aidial_adapter_openai.env import (
    GPT4_VISION_DEPLOYMENTS,
    GPT4O_DEPLOYMENTS,
    GPT4O_MINI_DEPLOYMENTS,
)
from aidial_adapter_openai.utils.image import ImageDetail, resolve_detail_level


class ImageTokenizer(BaseModel):
    low_detail_tokens: int
    """
    Number of tokens per image in low resolution mode
    """

    tokens_per_tile: int
    """
    Number of tokens per one tile image of 512x512 size
    """

    def tokenize(self, width: int, height: int, detail: ImageDetail) -> int:
        concrete_detail = resolve_detail_level(width, height, detail)
        match concrete_detail:
            case "low":
                return self.low_detail_tokens
            case "high":
                return self._compute_high_detail_tokens(width, height)
            case _:
                assert_never(concrete_detail)

    def _compute_high_detail_tokens(self, width: int, height: int) -> int:
        # Fit into 2048x2048 box
        width, height = _fit_longest(width, height, 2048)

        # Scale down so the shortest side is 768 pixels
        width, height = _fit_shortest(width, height, 768)

        # Calculate the number of 512-pixel tiles required
        cols = math.ceil(width / 512)
        rows = math.ceil(height / 512)

        return (self.tokens_per_tile * cols * rows) + self.low_detail_tokens


GPT4O_IMAGE_TOKENIZER = GPT4_VISION_IMAGE_TOKENIZER = ImageTokenizer(
    low_detail_tokens=85, tokens_per_tile=170
)
GPT4O_MINI_IMAGE_TOKENIZER = ImageTokenizer(
    low_detail_tokens=2833, tokens_per_tile=5667
)

_TOKENIZERS: List[Tuple[ImageTokenizer, List[str]]] = [
    (GPT4O_IMAGE_TOKENIZER, GPT4O_DEPLOYMENTS),
    (GPT4O_MINI_IMAGE_TOKENIZER, GPT4O_MINI_DEPLOYMENTS),
    (GPT4_VISION_IMAGE_TOKENIZER, GPT4_VISION_DEPLOYMENTS),
]


def get_image_tokenizer(deployment_id: str) -> ImageTokenizer | None:
    for tokenizer, ids in _TOKENIZERS:
        if deployment_id in ids:
            return tokenizer
    return None


def _fit_longest(width: int, height: int, size: int) -> tuple[int, int]:
    ratio = width / height
    if width > height:
        scaled_width = min(width, size)
        scaled_height = int(scaled_width / ratio)
    else:
        scaled_height = min(height, size)
        scaled_width = int(scaled_height * ratio)

    return scaled_width, scaled_height


def _fit_shortest(width: int, height: int, size: int) -> tuple[int, int]:
    ratio = width / height
    if width < height:
        scaled_width = min(width, size)
        scaled_height = int(scaled_width / ratio)
    else:
        scaled_height = min(height, size)
        scaled_width = int(scaled_height * ratio)

    return scaled_width, scaled_height
