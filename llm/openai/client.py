import requests
import logging

from llm.openai.models.chat import (
    ChatTool,
    ChatMessage,
    CreateChatRequest,
    CreateChatResponse,
)
from llm.openai.models.embeddings import (
    CreateEmbeddingRequest,
    Embedding,
    EmbeddingResponse,
)

log = logging.getLogger(__name__)


class Client:
    def __init__(self, token: str) -> None:
        self._create_chat_url = "https://api.openai.com/v1/chat/completions"
        self._create_embedding_url = "https://api.openai.com/v1/embeddings"

        self.token = token

    def send_chat(
        self, model: str, messages: list[ChatMessage], tools: list[ChatTool]
    ) -> CreateChatResponse:
        chat_req = CreateChatRequest(model=model, messages=messages, tools=tools)
        headers = {
            "content-type": "application/json",
            "Authorization": "Bearer " + self.token,
        }

        json = chat_req.model_dump_json(exclude_unset=True)
        log.debug(f"making api request with data: {json}")

        resp = requests.post(
            self._create_chat_url,
            headers=headers,
            data=json,
        )

        log.debug(f"recieved headers: {resp.headers}")
        json_res = resp.json()
        log.debug(f"recieved data: {json_res}")

        resp.raise_for_status()

        return CreateChatResponse(**json_res)

    def create_embedding(self, model: str, text: str) -> EmbeddingResponse:
        embedding_req = CreateEmbeddingRequest(input=text, model=model)

        headers = {
            "content-type": "application/json",
            "Authorization": "Bearer " + self.token,
        }

        json = embedding_req.model_dump_json(exclude_unset=True)
        log.debug(f"making api request with data: {json}")

        resp = requests.post(
            self._create_embedding_url,
            headers=headers,
            data=json,
        )

        log.debug(f"recieved headers: {resp.headers}")
        json_res = resp.json()
        log.debug(f"recieved data: {json_res}")

        resp.raise_for_status()

        return EmbeddingResponse(**json_res)
