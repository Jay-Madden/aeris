import datetime
import os
import random
from typing import Annotated

import domain.primitives.file as file
import domain.primitives.time as time
import domain.primitives.control as control

from llm.session import (
    Session,
    Param,
    SessionEndError,
    GPT3_5_FUNCTION,
    GPT4_FUNCTION,
    SessionResponseContext,
    GPT3_5_FUNCTION_16K,
    SessionGroup,
)

import pytz
from colorama import Fore, Style


def print_response(message: SessionResponseContext) -> None:
    print(
        f"{Fore.BLUE}Aeris {Style.RESET_ALL}{Style.DIM}({message.model}) {Style.NORMAL}{Fore.BLUE} >> {Style.RESET_ALL} {message.content}"
    )


if not (token := os.getenv("OPENAI_API_KEY")):
    raise Exception("'OPENAI_API_KEY not found")

session = Session(
    token=token, default_model=GPT3_5_FUNCTION_16K, response_callback=print_response
)


session.add_group(time.group)
session.add_group(file.group)
session.add_group(control.group)


def main() -> None:
    try:
        while True:
            text = input(f"{Fore.GREEN}User >> {Style.RESET_ALL}")
            session.make_request(text)
    except SessionEndError:
        pass
    except KeyboardInterrupt:
        pass

    print("\n-- Done --")


if __name__ == "__main__":
    main()
