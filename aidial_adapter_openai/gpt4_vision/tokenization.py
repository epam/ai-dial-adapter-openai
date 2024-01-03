"""
Tokenization of images as specified at
    https://learn.microsoft.com/en-us/azure/ai-services/openai/overview#image-tokens-gpt-4-turbo-with-vision
"""

import base64
import math
from io import BytesIO

from PIL import Image

from aidial_adapter_openai.gpt4_vision.messages import ImageDetail
from aidial_adapter_openai.utils.image_data_url import ImageDataURL


def tokenize_image(image: ImageDataURL, detail: ImageDetail) -> int:
    image_data = base64.b64decode(image.data)
    with Image.open(BytesIO(image_data)) as img:
        width, height = img.size
        return tokenize_image_by_size(width, height, detail)


def tokenize_image_by_size(width: int, height: int, detail: ImageDetail) -> int:
    is_low_detail = detail == "low" or (
        detail == "auto" and width <= 512 and height <= 512
    )

    return 85 if is_low_detail else compute_high_detail_tokens(width, height)


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
