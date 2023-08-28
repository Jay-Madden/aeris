import datetime
import random
from typing import Annotated

from llm.session import Session, Param

import pytz


def print_response(message: str) -> None:
    print(message)


session = Session(response_callback=print_response)


@session.function("Get the current time in UTC ISO-8601 format")
def get_current_time(tz_name: Annotated[str, Param(description="Olsen tz name of the timezone to get the current time of")]) -> str:
    tz = pytz.timezone(tz_name)
    return tz.fromutc(datetime.datetime.utcnow()).isoformat()


@session.function("Gets a random harry potter quote and appends it to the input string")
def get_harry_potter_quote() -> str:
    return random.choice(
        [
            "Happiness can be found, even in the darkest of times, if one only remembers to turn on the light",
            "Dobby is free",
            "Training for the ballet, Potter?",
            "He can run faster than Severus Snape confronted with shampoo",
        ]
    )


def main() -> None:
    """

    session.make_request(
        "call the random test function with a random sentence that sounds like its from harry potter and explain to me the result"
    )
    """
    # print(session.make_request("Set thermostat to 92 degress please"))
    session.make_request(input("User >> "))


if __name__ == "__main__":
    main()
