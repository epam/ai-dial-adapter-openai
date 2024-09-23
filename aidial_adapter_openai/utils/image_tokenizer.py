"""
Tokenization of images as specified at
    https://learn.microsoft.com/en-us/azure/ai-services/openai/overview#image-tokens-gpt-4-turbo-with-vision
"""

import base64
import math
from io import BytesIO
from typing import assert_never

from PIL import Image

from aidial_adapter_openai.utils.image import (
    ImageDataURL,
    ImageDetail,
    resolve_detail_level,
)


def tokenize_image_data_url(image_data: str, detail: ImageDetail) -> int:
    parsed_image_data = ImageDataURL.from_data_url(image_data)
    if not parsed_image_data:
        raise ValueError(f"Invalid image data URL {parsed_image_data!r}")
    return tokenize_image(parsed_image_data, detail)


def tokenize_image(image: ImageDataURL, detail: ImageDetail) -> int:
    image_data = base64.b64decode(image.data)
    with Image.open(BytesIO(image_data)) as img:
        width, height = img.size
        return tokenize_image_by_size(width, height, detail)


def tokenize_image_by_size(width: int, height: int, detail: ImageDetail) -> int:
    concrete_detail = resolve_detail_level(width, height, detail)
    match concrete_detail:
        case "low":
            return 85
        case "high":
            return compute_high_detail_tokens(width, height)
        case _:
            assert_never(concrete_detail)


def fit_longest(width: int, height: int, size: int) -> tuple[int, int]:
    ratio = width / height
    if width > height:
        scaled_width = min(width, size)
        scaled_height = int(scaled_width / ratio)
    else:
        scaled_height = min(height, size)
        scaled_width = int(scaled_height * ratio)

    return scaled_width, scaled_height


def fit_shortest(width: int, height: int, size: int) -> tuple[int, int]:
    ratio = width / height
    if width < height:
        scaled_width = min(width, size)
        scaled_height = int(scaled_width / ratio)
    else:
        scaled_height = min(height, size)
        scaled_width = int(scaled_height * ratio)

    return scaled_width, scaled_height


def compute_high_detail_tokens(width: int, height: int) -> int:
    # Fit into 2048x2048 box
    width, height = fit_longest(width, height, 2048)

    # Scale down so the shortest side is 768 pixels
    width, height = fit_shortest(width, height, 768)

    # Calculate the number of 512-pixel tiles required
    cols = math.ceil(width / 512)
    rows = math.ceil(height / 512)

    return (170 * cols * rows) + 85
