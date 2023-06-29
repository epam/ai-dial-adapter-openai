import vertexai
from vertexai.language_models._language_models import _TUNED_MODEL_LOCATION
from vertexai.preview.language_models import ChatModel

from utils.env import get_project_id
from utils.init import init
from utils.input import make_input
from utils.printing import print_ai


def main():
    # Currently chat-bison is only available in us-central1
    location = _TUNED_MODEL_LOCATION  # "us-central1"
    model_id = "chat-bison@001"

    vertexai.init(project=get_project_id(), location=location)
    chat_model = ChatModel.from_pretrained(model_id)
    parameters = {
        "temperature": 0.2,
        "max_output_tokens": 256,
        "top_p": 0.8,
        "top_k": 40,
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
