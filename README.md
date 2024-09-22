# Aeris 🧚 (Autonomous Engine for Real-time Intelligent Support)

## General AGI assistant framework/playground

Allows for easily adding and removing model integrated functions. And keeping track of a long stateful interaction with a given model. 

### Example

```python
from llm.session import Session, SessionGroup

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
