import json
from pathlib import Path

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account

from llm.vertex_ai_deployments import ChatCompletionDeployment
from utils.env import get_env
from utils.init import init

init()

JSON_CREDENTIALS = get_env("GOOGLE_APPLICATION_CREDENTIALS")
PROJECT_ID = get_env("GCP_PROJECT_ID")
LOCATION = get_env("DEFAULT_REGION")

API_ENDPOINT = "us-central1-aiplatform.googleapis.com"
MODEL_ID = ChatCompletionDeployment.CHAT_BISON_1.value

credentials = service_account.Credentials.from_service_account_file(
    JSON_CREDENTIALS,
    scopes=["https://www.googleapis.com/auth/cloud-platform"],
)

token_file = Path("secret") / ".gcp-token"

if not credentials.valid:
    print("Refreshing the token...")
    credentials.refresh(Request())
    token_file.write_text(credentials.token)

access_token = token_file.read_text()

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
print(json.dumps(response.json(), indent=2))

# # Token count
# # Needs PaLM API Key
# payload = {
#     "prompt": {
#         "messages": [
#             {"content": "How many tokens?"},
#             {"content": "For this whole conversation?"},
#         ]
#     }
# }

# url = f"https://generativelanguage.googleapis.com/v1beta2/models/chat-bison-001:countMessageTokens?key={API_KEY}"
# data = json.dumps(payload)
# response = requests.post(url, headers=headers, data=data)
# print(json.dumps(response.json(), indent=2))
