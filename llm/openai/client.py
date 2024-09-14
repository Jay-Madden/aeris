import os
import json

import requests

from llm.openai.models import (
    ChatFunction,
    ChatMessage,
    CreateChatRequest,
    CreateChatResponse,
)


class Client:
    def __init__(self, token: str) -> None:
        self.create_chat_url = "https://api.openai.com/v1/chat/completions"

        self.token = token

    def send_chat(
        self, model: str, messages: list[ChatMessage], functions: list[ChatFunction]
    ) -> CreateChatResponse:
        chat_req = CreateChatRequest(
            model=model, messages=messages, functions=functions
        )
        headers = {
            "content-type": "application/json",
            "Authorization": "Bearer " + self.token,
        }

        json = chat_req.model_dump_json(exclude_unset=True)
        resp = requests.post(
            self.create_chat_url,
            headers=headers,
            data=json,
        )

        resp.raise_for_status()

        json_res = resp.json()
        return CreateChatResponse(**json_res)
