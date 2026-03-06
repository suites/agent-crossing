from dataclasses import dataclass
from typing import Protocol

import numpy as np
from settings import EMBEDDING_DIMENSION


@dataclass(frozen=True)
class EmbeddingEncodingContext:
    text: str


class EmbeddingEncoder(Protocol):
    def encode(self, context: EmbeddingEncodingContext) -> np.ndarray: ...


class EmbeddingClient(Protocol):
    def embed(
        self,
        *,
        model: str | None = None,
        input: str,
        truncate: bool = True,
        keep_alive: str = "30m",
        expected_dimension: int | None = None,
    ) -> list[float]: ...


class LlmEmbeddingEncoder:
    def __init__(
        self,
        client: EmbeddingClient,
        model: str = "bge-m3",
    ) -> None:
        self.client: EmbeddingClient = client
        self.model: str = model

    def encode(self, context: EmbeddingEncodingContext) -> np.ndarray:
        vector = self.client.embed(
            model=self.model,
            input=context.text,
            truncate=True,
            keep_alive="30m",
            expected_dimension=EMBEDDING_DIMENSION,
        )
        return np.asarray(vector, dtype=np.float32)


OllamaEmbeddingEncoder = LlmEmbeddingEncoder
