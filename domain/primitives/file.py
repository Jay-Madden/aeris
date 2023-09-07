from typing import Annotated

from llm.session import SessionGroup, Param

group = SessionGroup()


@group.function(
    "Writes a file on the computer at a given path overwriting everything in the file"
)
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


@group.function("Reads a file on the computer at a given path")
def read_file(
    file_path: Annotated[
        str, Param(description="The relative path to the file to read")
    ]
) -> str:
    with open(file_path) as f:
        return f.read()
