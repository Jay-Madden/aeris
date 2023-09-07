from llm.session import SessionGroup, SessionEndError

group = SessionGroup()


@group.function("Ends the current chat thread")
def end_chat() -> None:
    raise SessionEndError()
