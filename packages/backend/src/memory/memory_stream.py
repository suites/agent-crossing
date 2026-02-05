import datetime

import numpy as np

from .memory_object import MemoryObject


class MemoryStream:
    def __init__(self):
        self.memories: list[MemoryObject] = []

    def add_memory(
        self,
        description: str,
        now: datetime.datetime,
        importance: int,
        embedding: np.ndarray,
    ):
        new_memory = MemoryObject(
            id=len(self.memories),
            description=description,
            creation_timestamp=now,
            last_accessed=now,
            importance=importance,
            embedding=embedding,
        )
        self.memories.append(new_memory)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        current_time: datetime.datetime,
        top_k: int = 3,
    ) -> list[MemoryObject]:
        return []

    def _calculate_recency_score(
        self, memory: MemoryObject, current_time: datetime.datetime
    ) -> float:
        return 0.0

    def _calculate_relevance_score(
        self, memory: MemoryObject, query_embedding: np.ndarray
    ) -> float:
        return 0.0

    def _normalize(self, scores: list[float]) -> list[float]:
        return []
