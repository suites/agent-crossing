from dataclasses import dataclass
from typing import Protocol

import numpy as np
from settings import EMBEDDING_DIMENSION

from .ollama_client import OllamaClient


@dataclass(frozen=True)
class EmbeddingEncodingContext:
    text: str


class EmbeddingEncoder(Protocol):
    def encode(self, context: EmbeddingEncodingContext) -> np.ndarray: ...


class OllamaEmbeddingEncoder:
    def __init__(
        self,
        client: OllamaClient,
    ) -> None:
        self.client: OllamaClient = client

    def encode(self, context: EmbeddingEncodingContext) -> np.ndarray:
        vector = self.client.embed(
            model="bge-m3",
            input=context.text,
            truncate=True,
            keep_alive="30m",
            expected_dimension=EMBEDDING_DIMENSION,
        )
        return np.asarray(vector, dtype=np.float32)
