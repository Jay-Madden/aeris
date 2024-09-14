from typing import Annotated

from llm.session import SessionGroup, Param

group = SessionGroup()

@group.function(
    "Gets the current state of the house and its rooms"
)
def get_house_state() -> str:
    return """
        +-----------------+-----------------+
        |                 |                 |
        |                 |                 |
        |                 |                 |
        | Room 1 (on)     \\ Room 2 (on)    |
        |                 |                 |
        |                 |                 |
        |                 |                 |
        +--------------------+              |
        |                    |              |
        |                    |              |
        |                    |              |
        |                   /               |
        |  Room 3 (off)      |              |
        |                    |              |
        |                    |              |
        |                    |              |
        +--------------------+--------------+    
    """

@group.function(
    "Turns the lights on or off in a given room number"
)
def control_room_light(
    room_number: Annotated[
        int, Param(description="The room number to target")
    ],
    state: Annotated[
        bool, Param(description="The state to set the light in the room too, must be one of [on|off]")
    ],
) -> bool:

    if state not in ("on", "off"):
        raise ValueError("State must be one of [on|off]")

    print(f"Room {room_number} light turned off {state}")

    return True
