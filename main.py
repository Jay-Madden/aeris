import datetime
import random
from typing import Annotated

from llm.session import (
    Session,
    Param,
    SessionEndError,
    GPT3_5_FUNCTION,
    GPT4_FUNCTION,
    SessionResponseContext,
)

import pytz
from colorama import Fore, Style


def print_response(message: SessionResponseContext) -> None:
    print(
        f"{Fore.BLUE}Aeris {Style.RESET_ALL}{Style.DIM}({message.model}) {Style.NORMAL}{Fore.BLUE} >> {Style.RESET_ALL} {message.content}"
    )


session = Session(default_model=GPT4_FUNCTION, response_callback=print_response)


@session.function("Ends the current chat thread")
def end_chat() -> None:
    raise SessionEndError()


@session.function("Writes a file on the computer at a given path")
def write_file(
    file_path: Annotated[
        str, Param(description="The relative path to the file to read")
    ],
    content: Annotated[
        str, Param(description="The content to write to the given file")
    ],
) -> None:
    with open(file_path, "w+") as f:
        f.write(content)


@session.function("Reads a file on the computer at a given path")
def read_file(
    file_path: Annotated[
        str, Param(description="The relative path to the file to read")
    ]
) -> str:
    with open(file_path) as f:
        return f.read()


@session.function("Get the current time in UTC ISO-8601 format")
def get_current_time(
    tz_name: Annotated[
        str,
        Param(description="Olsen tz name of the timezone to get the current time of"),
    ]
) -> str:
    tz = pytz.timezone(tz_name)
    return tz.fromutc(datetime.datetime.utcnow()).isoformat()


def main() -> None:
    try:
        while True:
            text = input("User >> ")
            session.make_request(text)
    except SessionEndError:
        pass
    except KeyboardInterrupt:
        pass

    print("\n-- Done --")


if __name__ == "__main__":
    main()
