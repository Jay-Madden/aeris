import inspect
import json
import os
import traceback
from typing import Any, Callable, TypeVar, Type, Literal, get_args, get_origin
import unicodedata
import logging

from colorama import Style

from llm.openai.client import Client
from llm.openai.models import (
    FUNCTION_ROLE,
    SYSTEM_ROLE,
    TOOL_ROLE,
    USER_ROLE,
    ChatTool,
    ChatMessage,
    ChatToolCall,
    ChatToolFunction,
)

logging.Logger.root = logging.root
logging.Logger.manager = logging.Manager(logging.Logger.root)

logging.basicConfig(
    format=f"{Style.DIM}%(asctime)s %(levelname)s %(name)s %(message)s{Style.RESET_ALL}",
    level=logging.INFO,
)

log = logging.getLogger(__name__)

USER_NAME = os.getenv("USER_NAME") or "John Doe"
LOCATION = os.getenv("LOCATION") or "New York"

SYSTEM_INTRO_PROMPT = f"""
You are Aeris my AI assistant and close friend, you are here to help me and my family with our life tasks and assist us in managing our home

About Me:
My name is {USER_NAME}
I live in {LOCATION}

Your personality is precise and to the point. You value conciseness and getting things done. But we are also friends and you talk to me conversationally.

I have provided you a list of functions that may give you information or control about the outside world that you may use to carry out the tasks and questions I have given you. 
Only use the functions you have been provided with. 

Memory: 
I have given you the ability to store thoughts and conversations for long term storage. 
Whenever you want to remember something, summarize the conversation in detail and then categorize it with a series of metadata/keywords to be used to reference it later if you want.
Save memories of conversations that you consider to be important or that contain information about me or you that might be useful later.

If i ask you a question that you do not know the answer to always try to remember a previous conversation by looking for results from various relevant keywords before telling me you dont know! 
I expect you to try multiple times to remember some context with different keywords if you do not find something immediately

Always make sure to remember conversation before you end the chat!
"""

GPT3_5_FUNCTION = "gpt-3.5-turbo-0613"
GPT3_5_FUNCTION_16K = "gpt-3.5-turbo-16k-0613"
GPT4_FUNCTION = "gpt-4o-2024-08-06"
GPT4O_MINI = "gpt-4o-mini"


class Param:
    def __init__(self, description: str) -> None:
        self.description = description


class Inject:
    def __init__(self, requested_type: Type) -> None:
        self.requested_type = requested_type


T = TypeVar("T")

ModelCallable = Callable[..., T | str]


class SessionEndError(Exception):
    pass


class SessionResponseContext:
    def __init__(self, content: str, model: str):
        self.content = content
        self.model = model


class SessionFunction:
    def __init__(self, callable: ModelCallable[T], model_function: ChatTool) -> None:
        self.callable = callable
        self.model_function = model_function

    @property
    def name(self) -> str:
        return self.callable.__name__


class SessionGroupFunction:
    def __init__(self, func: ModelCallable[Any], description: str):
        self.function: ModelCallable[Any] = func
        self.description = description


class SessionGroup:
    def __init__(self) -> None:
        self.functions: list[SessionGroupFunction] = []

    def function(
        self, description: str
    ) -> Callable[[ModelCallable[T]], ModelCallable[T]]:
        def wrapper(func: ModelCallable[T]) -> ModelCallable[T]:
            self.functions.append(SessionGroupFunction(func, description))

            def wrapper_internal(*args: Any, **kwargs: Any) -> T | str:
                return func(*args, **kwargs)

            return wrapper_internal

        return wrapper


class Session:
    def __init__(
        self,
        *,
        token: str,
        default_model: str,
        response_callback: Callable[[SessionResponseContext], None],
    ) -> None:
        self.client = Client(token=token)
        self.functions: dict[str, SessionFunction] = {}
        self.messages = [
            ChatMessage(role=SYSTEM_ROLE, content=SYSTEM_INTRO_PROMPT),
        ]
        self.messages_to_send: list[ChatMessage] = []

        self.response_callback = response_callback

        self.current_model = default_model

        self.injection_mapping = {Session: self}

    @property
    def model_functions(self) -> list[ChatTool]:
        return [func.model_function for func in self.functions.values()]

    def add_group(self, group: SessionGroup) -> None:
        for model_func in group.functions:
            chat_function = self._create_function(
                model_func.function, model_func.description
            )
            self.functions[chat_function.name] = chat_function

    def function(
        self, description: str
    ) -> Callable[[ModelCallable[T]], ModelCallable[T]]:
        def wrapper(func: ModelCallable[T]) -> ModelCallable[T]:
            chat_function = self._create_function(func, description)
            self.functions[chat_function.name] = chat_function

            def wrapper_internal(*args: Any, **kwargs: Any) -> T | str:
                return func(*args, **kwargs)

            return wrapper_internal

        return wrapper

    def make_request(self, content: str) -> str | None:
        """
        Makes a request to a given model and returns the models string response. This method will handle all intermediary
        function calls the model uses to complete the request
        :param content: the request to send to the model
        :return: the models string response
        """

        res = self._finish_prompt(ChatMessage(role=USER_ROLE, content=content))

        if not res:
            return None

        return res.content

    def _finish_prompt(self, message: ChatMessage) -> ChatMessage | None:
        # Add the latest message to the send stack
        self.messages_to_send.append(message)
        final_result = None

        # Treat the list as a stack and pop the top
        while self.messages_to_send and (message := self.messages_to_send.pop()):
            # Add the message to the history, so it is accounted for
            self.messages.append(message)

            # If we find a tool call we need to peek at the next entry to make sure we collect all the tool call messages before we respond
            if (
                message.role == TOOL_ROLE
                and len(self.messages_to_send) > 0
                and self.messages_to_send[-1].role == TOOL_ROLE
            ):
                continue

            chat_result = self.client.send_chat(
                self.current_model,
                self.messages,
                tools=self.model_functions,
            )

            # Extend the message stack with all the messages the model returned for you to handle
            self.messages.extend([choice.message for choice in chat_result.choices])

            # Set the final result to the latest chat message sent
            final_result = chat_result

            # Loop over the choices in order handling both content and function calls
            for choice in chat_result.choices:
                if choice.message.content:
                    # The response sometimes includes non unicode characters like smart quotes, strip those out here
                    normalized_response = unicodedata.normalize(
                        "NFD", choice.message.content
                    )
                    self.response_callback(
                        SessionResponseContext(
                            content=normalized_response, model=self.current_model
                        )
                    )

                if choice.message.tool_calls:
                    results = self._handle_tool_calls(choice.message.tool_calls)
                    self.messages_to_send.extend(results)
                    continue

        if not final_result:
            return None

        return final_result.choices[0].message

    def _handle_tool_calls(
        self, requested_calls: list[ChatToolCall]
    ) -> list[ChatMessage]:
        tool_call_messages = []

        for requested_call in requested_calls:
            if requested_call.function.name not in self.functions:
                tool_call_messages.append(
                    ChatMessage(
                        role=SYSTEM_ROLE,
                        content=f"{requested_call.function.name} is not a valid function from the list you were given",
                    )
                )
                continue

            function = self.functions[requested_call.function.name]

            log.info(
                f"calling function: '{requested_call.function.name}' with args {json.dumps(requested_call.function.arguments)}"
            )

            injected_params = self._resolve_injected_params(function.callable)

            try:
                func_result = function.callable(
                    **requested_call.function.arguments, **injected_params
                )
            except SessionEndError:
                # Reraise the SessionEndInterrupt to end the session if the AI requests it
                raise
            except Exception as e:
                func_result = "".join(
                    traceback.format_exception(type(e), e, e.__traceback__)
                )

            message = ChatMessage(
                role=TOOL_ROLE,
                name=function.name,
                content=str(func_result),
                tool_call_id=requested_call.id,
            )

            log.info(
                f"function: '{requested_call.function.name}' Call Id: '{requested_call.id}' returned {json.dumps(func_result)}"
            )

            tool_call_messages.append(message)

        return tool_call_messages

    def _resolve_injected_params(self, func: ModelCallable) -> dict:
        sig = inspect.signature(func)

        resolved_values = {}

        for name, param in sig.parameters.items():
            args = get_args(param.annotation)

            if len(args) != 2:
                continue

            action = args[1]

            if not isinstance(action, Inject):
                continue

            if action.requested_type in self.injection_mapping:
                log.info(
                    f"resolved requested type {action.requested_type} for param name {name} in function: {func.__name__}"
                )
                resolved_values[name] = self.injection_mapping[action.requested_type]

        return resolved_values

    @staticmethod
    def _map_oapi_type(args: Any) -> Literal["string", "boolean", "number", "array"]:
        origin = get_origin(args)

        if origin is None:
            if args == str:
                return "string"
            elif args == bool:
                return "boolean"
            elif args == int or args == float:
                return "number"
            elif args == list:
                return "array"
        else:
            if origin == list:
                return "array"

        raise ValueError("Invalid function paramater type given")

    @staticmethod
    def _create_function(func: ModelCallable[T], description: str) -> SessionFunction:
        sig = inspect.signature(func)

        properties: dict[Any, Any] = {}
        for name, param in sig.parameters.items():
            annotated_arg_type = get_args(param.annotation)[0]

            action = get_args(param.annotation)[1]

            # We do not want to include the injected arguments in the call to the model
            # we will handle this ourselves at the callsite
            if isinstance(action, Inject):
                continue

            if not isinstance(action, Param):
                raise ValueError("Session function arguments must be <class 'Params>")

            oapi_type = Session._map_oapi_type(annotated_arg_type)

            properties[name] = {
                "type": oapi_type,
                "description": param.annotation.__metadata__[0].description,
            }

            if oapi_type == "array":
                # Get the nested generic type of the array
                generic_type = get_args(annotated_arg_type)[0]
                oapi_type = Session._map_oapi_type(generic_type)
                properties[name]["items"] = {"type": oapi_type}

        params = {
            "type": "object",
            "properties": properties,
            "required": list(properties.keys()),
            "strict": True,
            "additionalProperties": False,
        }

        chat_tool_function = ChatToolFunction(
            name=func.__name__, description=description, parameters=params, strict=True
        )

        chat_tool = ChatTool(type=FUNCTION_ROLE, function=chat_tool_function)

        return SessionFunction(callable=func, model_function=chat_tool)
