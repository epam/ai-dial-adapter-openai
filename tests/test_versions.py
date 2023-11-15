from typing import Optional

import pytest

from aidial_adapter_openai.utils.versions import compare_versions

compare_versions_dataset = [
    ("2023-07-01", "2023-07-01", 0),
    ("2023-07-01", "2023-07-01-preview", 0),
    ("2023-07-01-preview", "2023-07-01", 0),
    ("2023-07-01-preview", "2023-07-01-preview", 0),
    ("2023-07-0", "2023-07-01", None),
    ("2022-12-01", "2023-06-01-preview", -1),
    ("2023-09-01-preview", "2023-05-15", 1),
]


@pytest.mark.parametrize("v1, v2, result", compare_versions_dataset)
def test_compare_versions(v1: str, v2: str, result: Optional[int]):
    assert compare_versions(v1, v2) == result
