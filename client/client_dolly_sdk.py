import json

from google.cloud.aiplatform import Endpoint, init

from utils.env import get_env
from utils.perf import Timer

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("DEFAULT_REGION")
ENDPOINT_ID = get_env("DOLLY2_ENDPOINT")

init(project=PROJECT_ID, location=REGION)

endpoint = Endpoint(
    f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}"
)

instances = [
    {"text": "Explain to me the difference between nuclear fission and fusion."}
]

with Timer("Prediction") as elapsed:
    response = endpoint.predict(instances=instances)
    print(json.dumps(response, indent=2))
    content = response[0][0]["generated_text"]

print(f"Time: {elapsed.time:.3f} sec")
print(f"#char: {len(content)}")
print(f"Rate: {len(content) / elapsed.time:.3f} char/sec")
