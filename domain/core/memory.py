import json
import random
from typing import Annotated, MutableMapping

from llm.openai.models import ChatMessage
from llm.session import Inject, Session, SessionGroup, Param

from pydantic import BaseModel

group = SessionGroup()


class Memory(BaseModel):
    summary: str
    keywords: list[str]
    conversation: list[ChatMessage]


@group.function("Store a memory and keyword list describing the memory")
def store_memory(
    detailed_summary: Annotated[
        str,
        Param(description="Detailed summary of the conversation this memory is about"),
    ],
    keywords: Annotated[
        list[str],
        Param(
            description="List of single words without a space to be associated with this memory"
        ),
    ],
    session: Annotated[Session, Inject(Session)],
) -> None:
    for kw in keywords:
        if " " in kw:
            raise ValueError("Keywords cannot contain a space")

    try:
        with open("model_output/memories.json", "r+") as f:
            memories = json.loads(f.read()) or {}
    except (FileNotFoundError, json.JSONDecodeError):
        memories = {}

    keywords = [kw.lower() for kw in keywords]

    mem_object = Memory(
        summary=detailed_summary, keywords=keywords, conversation=session.messages
    )
    memories[str(random.randint(0, 10000))] = mem_object.model_dump()

    with open("model_output/memories.json", "w+") as f:
        f.write(json.dumps(memories, indent=2))


@group.function("Recall a set of memories from a given list of keywords")
def recall_memory_keyword(
    keywords: Annotated[
        list[str],
        Param(description="List of keywords to use to search for a given memory"),
    ]
) -> str:
    for kw in keywords:
        if " " in kw:
            raise ValueError("Keywords cannot contain a space")

    try:
        with open("model_output/memories.json", "r+") as f:
            memories = json.loads(f.read()) or {}
    except FileNotFoundError:
        return ""

    memory_summaries: list[str] = []

    for memory in memories.values():
        for kw in keywords:
            if kw.lower() in [kw_mem.lower() for kw_mem in memory["keywords"]]:
                memory_summaries.append(memory["summary"])

    return (
        json.dumps(memory_summaries)
        if memory_summaries
        else "No results remembered please try again"
    )
