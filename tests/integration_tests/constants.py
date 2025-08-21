import logging
import os
from pathlib import Path

from aidial_adapter_openai.configuration.app_config import ApplicationConfig
from aidial_adapter_openai.utils.resource import Resource
from tests.integration_tests.base import TestDeployments

_log = logging.getLogger(__name__)

CURRENT_DIR = Path(__file__).parent

IMAGE_RESOURCE = Resource(
    type="image/png",
    data=(CURRENT_DIR / "assets" / "image.png").read_bytes(),
)

PDF_DOCUMENT_RESOURCE = Resource(
    type="application/pdf",
    data=(CURRENT_DIR / "assets" / "doc.pdf").read_bytes(),
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
    _log.warning(
        f"Cannot find the configuration file: {TEST_DEPLOYMENTS_CONFIG_PATH}. "
        "Using noop configuration."
    )
    TEST_DEPLOYMENTS_CONFIG = TestDeployments(
        deployments=[], app_config=ApplicationConfig()
    )
