import locust_plugins  # Needed to enable the locust plugins
from locust import HttpUser, between, task


class ChatUser(HttpUser):
    wait_time = between(1, 1.01)

    headers = {"Content-Type": "application/json"}

    @task
    def send_chat(self):
        url = "/openai/deployments/chat-bison%40001/chat/completions?api-version=2023-03-15-preview&project_id=vertex-ai-391108"
        data = {
            "messages": [{"role": "user", "content": "Compute 12+18"}],
            "model": "gpt-3.5-turbo",
        }
        self.client.post(url, headers=self.headers, json=data)
