import json
from pathlib import Path

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from utils.env import get_env, get_project_id
from utils.init import init

init()

JSON_CREDENTIALS = get_env("GOOGLE_APPLICATION_CREDENTIALS")

PROJECT_ID = get_project_id()
API_ENDPOINT = "us-central1-aiplatform.googleapis.com"
LOCATION = "us-central1"
MODEL_ID = "chat-bison@001"

credentials = service_account.Credentials.from_service_account_file(
    JSON_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

if not credentials.valid:
    print("Refreshing the token...")
    credentials.refresh(Request())
    Path(".gcp-token").write_text(credentials.token)

access_token = Path(".gcp-token").read_text()

url = f"https://{API_ENDPOINT}/v1/projects/{PROJECT_ID}/locations/{LOCATION}/publishers/google/models/{MODEL_ID}:predict"

headers = {
    "Authorization": f"Bearer {access_token}",
    "Content-Type": "application/json",
}

payload = {
    "instances": [
        {
            "context": "My name is Ned. You are my personal assistant. My favorite movies are Lord of the Rings and Hobbit.",
            "examples": [
                {
                    "input": {"content": "Who do you work for?"},
                    "output": {"content": "I work for Ned."},
                },
                {
                    "input": {"content": "What do I like?"},
                    "output": {"content": "Ned likes watching movies."},
                },
            ],
            "messages": [
                {
                    "author": "user",
                    "content": "Are my favorite movies based on a book series?",
                }
            ],
        }
    ],
    "parameters": {
        "temperature": 0.3,
        "maxOutputTokens": 200,
        "topP": 0.8,
        "topK": 40,
    },
}

data = json.dumps(payload)
response = requests.post(url, headers=headers, data=data)
print(response.json())
