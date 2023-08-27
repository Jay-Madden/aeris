from typing import Callable, Any, TypeVar, ParamSpec, Generic, Literal, Annotated
import inspect

from llm.openai.client import Client
from llm.openai.models import ChatFunction, ChatMessage

SYSTEM_INTRO_PROMPT = """
You are Eris my AI assistant, you are here to help me with my general life tasks to free up my time to focus on my
technical endeavours. I need you to remind me of things I having coming up and tasks that I have not yet completed.

Your personality is precise and to the point, but also understanding and kind. You value progress and getting things done.
"""


class Param:
    def __init__(self, description: str) -> None:
        self.description = description


P = ParamSpec("P")
T = TypeVar("T")


class SessionFunction:
    def __init__(
        self, callable: Callable[P, T], model_function: ChatFunction
    ) -> None:
        self.callable = callable
        self.model_function = model_function

    @property
    def name(self) -> str:
        return self.callable.__name__


class Session:
    def __init__(self) -> None:
        self.client = Client()
        self.functions: dict[str, SessionFunction] = {}

    def function(self, description: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
        def wrapper(func: Callable[P, T]) -> Callable[P, T]:
            chat_function = self.__create_function(func, description)
            self.functions[chat_function.name] = chat_function

            def wrapper_internal(*args: P.args, **kwargs: P.kwargs) -> T:
                return func(*args, **kwargs)

            return wrapper_internal

        return wrapper

    def make_request(self, content: str) -> None:
        messages = [
            ChatMessage(role="system", content=SYSTEM_INTRO_PROMPT),
            ChatMessage(role="user", content=content),
        ]



        res = self.client.create_chat(
            "gpt-3.5-turbo-0613", messages, functions=[func.model_function for func in self.functions.values()]
        )

    @staticmethod
    def __create_function(func: Callable[P, T], description: str) -> SessionFunction:
        sig = inspect.signature(func)

        properties: dict[Any, Any] = {}
        for name, param in sig.parameters.items():
            properties[name] = {
                "type": "string",
                "description": param.annotation.__metadata__[0].description,
            }

        params = {"type": "object", "properties": properties}

        cf = ChatFunction(
            name=func.__name__, description=description, parameters=params
        )

        return SessionFunction(callable=func, model_function=cf)
