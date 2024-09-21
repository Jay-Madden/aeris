import json
from typing import Annotated

from llm.session import SessionGroup, Param

group = SessionGroup()


@group.function("Gets the current state of the house and its rooms")
def get_house_state() -> str:
    room_state = {
        1: {"light_state": False},
        2: {"light_state": False},
        3: {"light_state": True},
    }

    return json.dumps(room_state)


@group.function("Turns the lights on or off in a given room number")
def control_room_light(
    room_number: Annotated[int, Param(description="The room number to target")],
    state: Annotated[
        bool,
        Param(description="The state to set the light in the room too"),
    ],
) -> bool:
    if not isinstance(state, bool):
        raise ValueError("State must be a boolean")

    print(f"Room {room_number} light turned off {state}")

    return True
