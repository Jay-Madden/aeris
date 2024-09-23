import json
from os import stat
from typing import Annotated

from llm.session import SessionGroup, Param

group = SessionGroup()


@group.function("Gets the current state of the house and its rooms")
def get_house_state() -> str:
    with open("model_output/room_state.json", "r+") as f:
        rooms = json.loads(f.read()) or {}

    return json.dumps(rooms)


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

    with open("model_output/room_state.json", "r+") as f:
        rooms = json.loads(f.read()) or {}

    rooms[str(room_number)]["light_state"] = state

    with open("model_output/room_state.json", "w+") as f:
        f.write(json.dumps(rooms, indent=2))

    return True
