import datetime
import json
from typing import Any, Generic, Literal, ParamSpec

from pydantic import BaseModel, field_serializer, field_validator

ChatUserType = (
    Literal["user"] | Literal["system"] | Literal["assistant"] | Literal["function"]
)

P = ParamSpec("P")


class ChatFunctionCall(BaseModel):
    name: str
    arguments: dict[str, Any]

    # ---- OpenAI For some reason expects arguments to be in stringified object form, handle that here ----
    @field_serializer("arguments")
    def serialize_args(self, args: dict[str, Any]) -> str:
        return json.dumps(args)

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_args(cls, data: Any) -> Any:
        return json.loads(data)

    # --------


class ChatFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[Any, Any]


class ChatMessage(BaseModel):
    role: ChatUserType
    content: str | None = None
    name: str | None = None
    function_call: ChatFunctionCall | None = None


class CreateChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    functions: list[ChatFunction]
    function_call: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    n: int | None = None
    max_tokens: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    stream: bool | None = None
    stop: list[str] | None = None
    logit_bias: dict[str, int] | None = None
    user: str | None = None


class CreateChatChoices(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class CreateChatResponse(BaseModel):
    id: str
    object: str
    created: datetime.datetime
    model: str
    choices: list[CreateChatChoices]
