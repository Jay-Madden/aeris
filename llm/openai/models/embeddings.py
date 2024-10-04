from typing import Literal
from pydantic import BaseModel


class CreateEmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str
    encoding_format: Literal["float", "base64"] | None = None
    dimensions: int | None = None
    user: str | None = None


class Embedding(BaseModel):
    index: int
    embedding: list[float]
    object: Literal["embedding"]


class EmbeddingResponse(BaseModel):
    object: Literal["list"]
    data: list[Embedding]
