import json
import random
from typing import Annotated, MutableMapping

from llm.openai.models.chat import ChatMessage
from llm.session import TEXT_EMBEDDING_3_LARGE, Inject, Session, SessionGroup, Param
from llm.openai.client import Client

from pydantic import BaseModel
import psycopg

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
    client: Annotated[Client, Inject(Client)],
) -> None:
    for kw in keywords:
        if " " in kw:
            raise ValueError("Keywords cannot contain a space")

    keywords = [kw.lower() for kw in keywords]

    embedding = client.create_embedding(TEXT_EMBEDDING_3_LARGE, detailed_summary)

    conversation = ""
    for message in session.messages:
        if not message.role == "user" and not message.role == "assistant":
            continue

        if message.content:
            conversation += f"{message.role}: {message.content}\n"

    save_memory_db(embedding.data[0].embedding, detailed_summary, conversation)


# @group.function("Recall a memory based on a query sentence")
# def recall_memory(
#     query: Annotated[
#         str,
#         Param(
#             description="A short sentence describing what you are trying to remember"
#         ),
#     ],
#     client: Annotated[Client, Inject(Client)],
#     session: Annotated[Session, Inject(Session)],
# ) -> list[str]:
#     embedding = client.create_embedding(TEXT_EMBEDDING_3_LARGE, query)
#
#     with psycopg.connect("dbname=aeris_memory user=jaymadden") as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 "SELECT complete FROM memory ORDER BY embedding <-> %s::vector LIMIT 3",
#                 (embedding.data[0].embedding,),
#             )
#             conversations = cur.fetchall()
#
#     if not conversations:
#         return ["nothing to remember"]


@group.function("Recall a memory by its id")
def recall_memory_by_id(
    id: Annotated[
        int,
        Param(description="The id of the memory to remember"),
    ],
    client: Annotated[Client, Inject(Client)],
    session: Annotated[Session, Inject(Session)],
) -> str:
    with psycopg.connect("dbname=aeris_memory user=jaymadden") as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT complete FROM memory WHERE id = %s", (id,))
            conversation = cur.fetchone()

    if not conversation:
        return "nothing to remember"

    return conversation[0]


@group.function(
    "Recall a set of memories with an id based on a query sentence, to remember the full context call the memory_by_id function",
)
def recall_memory(
    query: Annotated[
        str,
        Param(
            description="A short sentence describing what you are trying to remember"
        ),
    ],
    client: Annotated[Client, Inject(Client)],
    session: Annotated[Session, Inject(Session)],
) -> list[str]:
    embedding = client.create_embedding(TEXT_EMBEDDING_3_LARGE, query)

    with psycopg.connect("dbname=aeris_memory user=jaymadden") as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id,summary FROM memory ORDER BY embedding <-> %s::vector LIMIT 3",
                (embedding.data[0].embedding,),
            )
            conversations = cur.fetchall()

    if not conversations:
        return ["nothing to remember"]

    return [
        f"id: {conversation[0]} summary: {conversation[1]}"
        for conversation in conversations
    ]


def save_memory_db(
    embedding: list[float], summary: str, complete: str, parent: int | None = None
):
    with psycopg.connect("dbname=aeris_memory user=jaymadden") as conn:
        with conn.cursor() as cur:
            if not parent:
                cur.execute(
                    "INSERT INTO memory (embedding, summary, complete) VALUES (%s, %s, %s)",
                    (embedding, summary, complete),
                )
            else:
                cur.execute(
                    "INSERT INTO memory (embedding, summary, complete, parent_memory_id) VALUES (%s, %s, %s)",
                    (embedding, summary, complete),
                )
        conn.commit()
