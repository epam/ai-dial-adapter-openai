from enum import Enum


class ChatCompletionDeployment(str, Enum):
    CHAT_BISON_1 = "chat-bison@001"
    CODECHAT_BISON_1 = "codechat-bison@001"

    def get_model_id(self) -> str:
        return self.value


class ExtraChatCompletionDeployment(str, Enum):
    DOLLY_V2_7B = "dolly-v2-7b"
    LLAMA2_7B_CHAT_1 = "llama2-7b-chat-001"


GECKO_MODEL_1 = "textembedding-gecko@001"


class EmbeddingsDeployment(str, Enum):
    TEXT_EMBEDDING_GECKO_1 = GECKO_MODEL_1
    TEXT_EMBEDDING_GECKO_1_CLASSIFICATION = f"{GECKO_MODEL_1}-classification"
    TEXT_EMBEDDING_GECKO_1_CLUSTERING = f"{GECKO_MODEL_1}-clustering"

    def get_model_id(self) -> str:
        return GECKO_MODEL_1
