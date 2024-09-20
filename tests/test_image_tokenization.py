from typing import List, Tuple

import pytest

from aidial_adapter_openai.utils.image_tokenizer import tokenize_image_by_size
from aidial_adapter_openai.utils.message_content_part import ImageDetail

test_cases: List[Tuple[int, int, ImageDetail, int]] = [
    (1, 1, "auto", 85),
    (1, 1, "high", 170 * 1 + 85),
    (100, 100, "low", 85),
    (500, 500, "high", 170 * 1 + 85),
    (512, 512, "auto", 85),
    (513, 513, "auto", 170 * 4 + 85),
    (2048, 4096, "high", 170 * 6 + 85),
    (512, 511, "auto", 85),
    (512, 513, "auto", 170 * 2 + 85),
    (768, 2048, "auto", 170 * 8 + 85),
    (768, 2050, "auto", 170 * 8 + 85),
    (800, 2050, "auto", 170 * 8 + 85),
    (10_000, 10_000, "auto", 170 * 4 + 85),
    (100_000, 100_000, "auto", 170 * 4 + 85),
]


@pytest.mark.parametrize("width, height, detail, expected_tokens", test_cases)
def test_tokenize(width, height, detail, expected_tokens):
    tokens1, detail1 = tokenize_image_by_size(width, height, detail)
    tokens2, detail2 = tokenize_image_by_size(height, width, detail)

    assert tokens1 == expected_tokens
    assert tokens2 == expected_tokens

    expected_detail = "low" if expected_tokens == 85 else "high"

    assert detail1 == expected_detail
    assert detail2 == expected_detail
