import json

from google.cloud.aiplatform import Endpoint, init

from llm.llama2 import get_llama2_chat_prompt
from utils.env import get_env
from utils.perf import Timer

PROJECT_ID = get_env("GCP_PROJECT_ID")
REGION = get_env("DEFAULT_REGION")
ENDPOINT_ID = get_env("LLAMA2_ENDPOINT")

init(project=PROJECT_ID, location=REGION)

endpoint = Endpoint(
    f"projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}"
)


chat_prompt = get_llama2_chat_prompt(
    system_prompt="You are a helpful assistant",
    chat_history=[("My name is Anton", "Hello! How can I assist you today?")],
    message="What's my name? Also write a poem about Valencia.",
)

completion_prompt = "Write a poem about Valencia."

instances = [
    {"prompt": chat_prompt, "max_length": 1024, "temperature": 1.0, "n": 1},
]

with Timer("Prediction") as elapsed:
    response = endpoint.predict(instances=instances)
    print(json.dumps(response, indent=2))
    content = response[0][0][0]["generated_text"]

print(f"Time: {elapsed.time:.3f} sec")
print(f"#char: {len(content)}")
print(f"Rate: {len(content) / elapsed.time:.3f} char/sec")
