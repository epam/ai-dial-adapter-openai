import json

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from utils.env import get_env
from utils.init import init
from utils.perf import Timer

init()

JSON_CREDENTIALS = get_env("GOOGLE_APPLICATION_CREDENTIALS")

API_ENDPOINT = "us-central1-aiplatform.googleapis.com"
LOCATION = get_env("DEFAULT_REGION")
PROJECT_ID = get_env("GCP_PROJECT_ID")
ENDPOINT_ID = get_env("LLAMA2_ENDPOINT")

credentials = service_account.Credentials.from_service_account_file(
    JSON_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

print("Refreshing the token...")
credentials.refresh(Request())
access_token = credentials.token

url = f"https://{API_ENDPOINT}/v1/projects/{PROJECT_ID}/locations/us-central1/endpoints/{ENDPOINT_ID}:predict"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

payload = {
    "instances": [
        {
            "prompt": "Write a poem about Valencia.",
            "max_length": 200,
            "top_k": 10,
            "n": 1,
        },
    ]
}
data = json.dumps(payload)

with Timer("Prediction") as elapsed:
    response = requests.post(url, headers=headers, data=data).json()
    print(json.dumps(response, indent=2))
    content = response["predictions"][0][0]["generated_text"]

print(f"Time: {elapsed.time:.3f} sec")
print(f"#char: {len(content)}")
print(f"Rate: {len(content) / elapsed.time:.3f} char/sec")
