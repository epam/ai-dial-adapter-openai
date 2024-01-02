import pytest

from aidial_adapter_openai.gpt4_vision.tokenization import (
    tokenize_image_by_size,
)

test_cases = [
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
]


@pytest.mark.parametrize("width, height, detail, expected", test_cases)
def test_tokenize(width, height, detail, expected):
    assert tokenize_image_by_size(width, height, detail) == expected
    assert tokenize_image_by_size(height, width, detail) == expected
