# Aeris 🧚 -- Autonomous Engine for Real-time Intelligent Support

## General AGI assistant framework/playground

A fastapi inspired lightweight llm agent framework.

Allows for easily adding and removing model integrated functions. And keeping track of a long stateful interaction with a given model. 

### Example

```python
from llm.session import Session, SessionGroup

group = SessionGroup()

@group.function("Reads a file on the computer at a given path")
def read_file(
    file_path: Annotated[
        str, Param(description="The relative path to the file to read")
    ]
) -> str:
    with open(file_path) as f:
        return f.read()


session = Session(
    token=token, default_model="gpt-4o-mini"
)

session.add_group(group)

def main() -> None:
    try:
        while True:
            text = input("User >> ")
            session.make_request(text)
    except SessionEndError:
        pass
    except KeyboardInterrupt:
        pass

```

Maybe one day this will actually be useful :laughing:

### Configuration

| Name           | Description                                                                              |
|----------------|------------------------------------------------------------------------------------------|
| OPENAI_API_KEY | Your openai api key                                                                      |
| USER_NAME      | The name of the primary person interacting with the assistant (Defaults to john doe)     |
| LOCATION       | The primary location of the person interacting with the assistant (Defaults to new york) |
