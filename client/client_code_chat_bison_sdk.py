import vertexai
from vertexai.preview.language_models import CodeChatModel

from llm.vertex_ai_deployments import ChatCompletionDeployment
from utils.env import get_env
from utils.init import init
from utils.input import make_input
from utils.printing import print_ai


def main():
    location = get_env("DEFAULT_REGION")
    project = get_env("GCP_PROJECT_ID")

    model_id = ChatCompletionDeployment.CODECHAT_BISON_1

    vertexai.init(project=project, location=location)
    chat_model = CodeChatModel.from_pretrained(model_id)
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 1024,
    }
    chat = chat_model.start_chat(**parameters)

    input = make_input()

    while True:
        content = input()
        response = chat.send_message(content)
        print_ai(response.text)


if __name__ == "__main__":
    init()
    main()
