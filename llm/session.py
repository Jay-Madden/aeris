import inspect
import os
from typing import Any, Callable, TypeVar

from llm.openai.client import Client
from llm.openai.models import ChatFunction, ChatFunctionCall, ChatMessage

USER_NAME = os.getenv("USER_NAME") or "John Doe"
LOCATION = os.getenv("LOCATION") or "New York"

SYSTEM_INTRO_PROMPT = f"""
You are Aeris my AI assistant, you are here to help me with my general life tasks to free up my time to focus on my
technical endeavours. 

About Me:
My name is {USER_NAME}
I live in {LOCATION}


Your personality is precise and to the point. You value conciseness and getting things done.

I have provided you a list of functions that may give you information or control about the outside world that you may use to carry out the tasks and questions I have given you. 
Only use the functions you have been provided with. 
Tell me if a task you have been given requires a function you have not been provided with.
"""

GPT3_5_FUNCTION = "gpt-3.5-turbo-0613"
GPT3_5_FUNCTION_16K = "gpt-3.5-turbo-16k-0613"
GPT4_FUNCTION = "gpt-4-0613"

CURRENT_MODEL = GPT3_5_FUNCTION_16K


class Param:
    def __init__(self, description: str) -> None:
        self.description = description


T = TypeVar("T")


ModelCallable = Callable[[Any], T] | Callable[[], T]


class SessionFunction:
    def __init__(
        self, callable: ModelCallable[T], model_function: ChatFunction
    ) -> None:
        self.callable = callable
        self.model_function = model_function

    @property
    def name(self) -> str:
        return self.callable.__name__


class Session:
    def __init__(self, *, response_callback: Callable[[str], None]) -> None:
        self.client = Client()
        self.functions: dict[str, SessionFunction] = {}
        self.messages = [
            ChatMessage(role="system", content=SYSTEM_INTRO_PROMPT),
        ]
        self.message_to_send: list[ChatMessage] = []

        self.response_callback = response_callback

    @property
    def model_functions(self) -> list[ChatFunction]:
        return [func.model_function for func in self.functions.values()]

    def function(
        self, description: str
    ) -> Callable[[ModelCallable[T]], ModelCallable[T]]:
        def wrapper(func: ModelCallable[T]) -> ModelCallable[T]:
            chat_function = self.__create_function(func, description)
            self.functions[chat_function.name] = chat_function

            def wrapper_internal(*args: Any, **kwargs: Any) -> T:
                return func(*args, **kwargs)

            return wrapper_internal

        return wrapper

    def make_request(self, content: str) -> str:
        res = self.__finish_prompt(ChatMessage(role="user", content=content))
        return res.content

    def __finish_prompt(self, message: ChatMessage) -> ChatMessage | None:
        self.message_to_send.append(message)
        final_result = None

        # Treat the list as a stack and pop the top
        while self.message_to_send and (message := self.message_to_send.pop()):
            # Add the message to the history, so it is accounted for
            self.messages.append(message)

            chat_result = self.client.send_chat(
                CURRENT_MODEL,
                self.messages + [message],
                functions=self.model_functions,
            )

            self.messages.extend([choice.message for choice in chat_result.choices])

            # Set the final result to the latest chat message sent
            final_result = chat_result

            # Loop over the choices in order handling both content and function calls
            for choice in chat_result.choices:
                if choice.message.content:
                    self.response_callback(choice.message.content)

                if choice.message.function_call:
                    results = self.__handle_function_calls(choice.message.function_call)
                    self.message_to_send.append(results)
                    continue

        if not final_result:
            return None

        return final_result.choices[0].message

    def __handle_function_calls(self, requested_call: ChatFunctionCall) -> ChatMessage:
        if requested_call.name in self.functions:
            function = self.functions[requested_call.name]
            func_args = requested_call.arguments.values()
            func_result = function.callable(*func_args)
            message = ChatMessage(
                role="function", name=function.name, content=str(func_result)
            )
        else:
            message = ChatMessage(
                role="system",
                content=f" {requested_call.name} is not a valid function from the list i gave you",
            )

        return message

    @staticmethod
    def __create_function(func: ModelCallable[T], description: str) -> SessionFunction:
        sig = inspect.signature(func)

        properties: dict[Any, Any] = {}
        for name, param in sig.parameters.items():
            properties[name] = {
                "type": "string",
                "description": param.annotation.__metadata__[0].description,
            }

        params = {
            "type": "object",
            "properties": properties,
            "required": [name for name, _ in sig.parameters.items()],
        }

        cf = ChatFunction(
            name=func.__name__, description=description, parameters=params
        )

        return SessionFunction(callable=func, model_function=cf)
