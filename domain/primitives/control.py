import time
from typing import Annotated
from llm.session import Param, SessionGroup, SessionEndError

group = SessionGroup()


@group.function("Ends the current chat thread")
def end_chat(
    memory_saved: Annotated[
        bool, Param(description="If this conversation has already been saved to memory")
    ]
) -> None:
    if not memory_saved:
        raise ValueError("All conversations must be remembered")

    raise SessionEndError()


@group.function("Pauses time for a set amount of seconds")
def wait(seconds: Annotated[int, Param(description="The time in seconds to wait")]):
    time.sleep(seconds)
