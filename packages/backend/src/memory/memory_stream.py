import datetime
from typing import List, Optional

import numpy as np

from .memory_object import MemoryObject, NodeType


class MemoryStream:
    def __init__(self):
        self.memories: list[MemoryObject] = []

    def add_memory(
        self,
        node_type: NodeType,
        citations: Optional[List[int]],
        description: str,
        now: datetime.datetime,
        importance: int,
        embedding: np.ndarray,
    ):
        """
        새로운 관찰(Observation)이나 생각(Reflection)을 스트림에 추가한다.
        """
        new_memory = MemoryObject(
            id=len(self.memories),
            node_type=node_type,
            citations=citations,
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
        """
        현재 상황에 가장 적합한 기억을 검색한다.

        """
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
