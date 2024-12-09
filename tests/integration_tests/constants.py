from pathlib import Path

from aidial_adapter_openai.utils.resource import Resource

CURRENT_DIR = Path(__file__).parent
SAMPLE_DOG_IMAGE_PATH = CURRENT_DIR / "images" / "dog-sample-image.png"
SAMPLE_DOG_RESOURCE = Resource(
    type="image/png",
    data=SAMPLE_DOG_IMAGE_PATH.read_bytes(),
)
