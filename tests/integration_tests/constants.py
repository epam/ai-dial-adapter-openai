import os
from pathlib import Path

from aidial_adapter_openai.app_config import ApplicationConfig
from aidial_adapter_openai.utils.resource import Resource
from tests.integration_tests.base import TestDeployments

CURRENT_DIR = Path(__file__).parent
SAMPLE_DOG_IMAGE_PATH = CURRENT_DIR / "images" / "dog-sample-image.png"
SAMPLE_DOG_RESOURCE = Resource(
    type="image/png",
    data=SAMPLE_DOG_IMAGE_PATH.read_bytes(),
)
TEST_DEPLOYMENTS_CONFIG_PATH = os.getenv(
    "INTEGRATION_TEST_DEPLOYMENTS_CONFIG_PATH",
    "tests/integration_tests/integration_test_config.json",
)
try:
    TEST_DEPLOYMENTS_CONFIG = TestDeployments.from_config(
        TEST_DEPLOYMENTS_CONFIG_PATH
    )
except FileNotFoundError:
    TEST_DEPLOYMENTS_CONFIG = TestDeployments(
        deployments=[], app_config=ApplicationConfig()
    )
