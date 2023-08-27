import datetime
import json
import os
from typing import Annotated

import openai

from llm.session import Session, Param

openai.api_key = os.getenv("OPENAI_API_KEY")

CURRENT_MODEL = "gpt-3.5-turbo-0613"


session = Session()


# @session.function("Function to get the current time in UTC ISO-8601 format")
def get_current_time() -> str:
    return datetime.datetime.utcnow().isoformat()


@session.function("Function to test random stuff out")
def test_func(s: Annotated[str, Param(description="SomeCoolDesc")]) -> None:
    print(s)


def get_dogs_names() -> str:
    return json.dumps(["hermione", "luna"])


FUNCTION_MAP = {"get_current_time": get_current_time, "get_dogs_names": get_dogs_names}

session.make_request(
    "Call the random test function with a random sentence that sounds like its from harry potter"
)


def main() -> None:
    messages = [
  #      {"role": "system", "content": SYSTEM_INTRO_PROMPT},
        {"role": "user", "content": "What are the names of my dogs and what "},
    ]
    functions = [
        {
            "name": "get_current_time",
            "description": "Gets the current time in UTC ISO-8601 format",
            "parameters": {"type": "object", "properties": {}},
        },
        {
            "name": "get_dogs_names",
            "description": "Gets the names of my dogs",
            "parameters": {"type": "object", "properties": {}},
        },
    ]

    chat_completion = openai.ChatCompletion.create(
        model=CURRENT_MODEL,
        functions=functions,
        messages=messages,
    )
    # print(json.dumps(chat_completion, indent=2))

    response_message = chat_completion["choices"][0]["message"]

    if response_message.get("function_call"):
        func = FUNCTION_MAP[response_message["function_call"]["name"]]
        func_res = func()
        messages.append(
            {
                "role": "function",
                "name": response_message["function_call"]["name"],
                "content": func_res,
            }
        )

        resp = openai.ChatCompletion.create(
            model=CURRENT_MODEL,
            functions=functions,
            messages=messages,
        )
        print(json.dumps(resp["choices"][0]["message"]["content"], indent=2))


if __name__ == "__main__":
    pass
    # main()
