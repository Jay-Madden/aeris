import datetime
import json
from typing import Any, Literal
from enum import Enum

from pydantic import BaseModel, field_serializer, field_validator

ChatRoleUser = Literal["user"]
USER_ROLE: ChatRoleUser = "user"

ChatRoleSystem = Literal["system"]
SYSTEM_ROLE: ChatRoleSystem = "system"

ChatRoleAssistant = Literal["assistant"]
ASSISTANT_ROLE: ChatRoleAssistant = "assistant"

ChatRoleFunction = Literal["function"]
FUNCTION_ROLE: ChatRoleFunction = "function"

ChatRoleTool = Literal["tool"]
TOOL_ROLE: ChatRoleTool = "tool"

ChatRoleType = (
    ChatRoleUser | ChatRoleSystem | ChatRoleAssistant | ChatRoleFunction | ChatRoleTool
)


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


class ChatToolFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[Any, Any]
    strict: bool = False


class ChatToolCall(BaseModel):
    id: str
    type: ChatRoleType
    function: ChatFunctionCall


class ChatTool(BaseModel):
    type: ChatRoleType
    function: ChatToolFunction


class ChatToolChoiceFunction(BaseModel):
    name: str


class ChatToolChoice(BaseModel):
    type: ChatRoleType
    name: ChatToolChoiceFunction


class ChatFunction(BaseModel):
    name: str
    description: str | None = None
    parameters: dict[Any, Any]


class ChatMessage(BaseModel):
    role: ChatRoleType
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ChatToolCall] | None = None


class CreateChatRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    functions: list[ChatFunction] | None = None
    function_call: str | None = None
    tools: list[ChatTool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ChatToolChoice | None = None
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


class CreateChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CreateChatResponse(BaseModel):
    id: str
    object: str
    created: datetime.datetime
    model: str
    choices: list[CreateChatChoices]
    usage: CreateChatUsage
