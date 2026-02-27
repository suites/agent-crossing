import datetime
from dataclasses import dataclass
from enum import Enum

import numpy as np
from llm import ImportanceScorer, ImportanceScoringContext, clamp_importance
from llm.embedding_encoder import EmbeddingEncodingContext, OllamaEmbeddingEncoder
from llm.llm_service import InsightWithCitation

from .memory_object import MemoryObject, NodeType
from .memory_stream import MemoryStream


class OrderBy(Enum):
    ASC = "ASC"
    DESC = "DESC"


@dataclass(frozen=True)
class ObservationContext:
    agent_name: str
    identity_stable_set: list[str]
    current_plan: str | None = None


class MemoryService:
    def __init__(
        self,
        *,
        memory_stream: MemoryStream,
        importance_scorer: ImportanceScorer,
        embedding_encoder: OllamaEmbeddingEncoder,
    ):
        self.memory_stream: MemoryStream = memory_stream
        self.importance_scorer: ImportanceScorer = importance_scorer
        self.embedding_encoder: OllamaEmbeddingEncoder = embedding_encoder

    def get_recent_memories(
        self,
        *,
        limit: int | None = None,
        order_by: OrderBy = OrderBy.DESC,
    ) -> list[MemoryObject]:
        """
        최근 메모리를 반환한다.
        - limit이 주어지면 상위 limit개까지만 반환한다.
        """
        sorted_memories = sorted(
            self.memory_stream.memories,
            key=lambda x: x.created_at,
            reverse=order_by == OrderBy.DESC,
        )

        if limit is not None:
            sorted_memories = sorted_memories[:limit]
        return sorted_memories

    def get_retrieval_memories(
        self,
        query: str,
        *,
        current_time: datetime.datetime,
        top_k: int = 3,
    ) -> list[MemoryObject]:
        """
        검색 쿼리를 기반으로 관련 메모리를 반환한다.
        """
        query_embedding = self.embedding_encoder.encode(
            EmbeddingEncodingContext(text=query)
        )

        return self.memory_stream.retrieve(
            query_embedding=query_embedding,
            top_k=top_k,
            current_time=current_time,
        )

    def create_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray,
        context: ObservationContext,
        importance: int | None = None,
    ) -> MemoryObject:
        final_importance = importance
        if final_importance is None:
            scoring_context = ImportanceScoringContext(
                observation=content,
                agent_name=context.agent_name,
                identity_stable_set=context.identity_stable_set,
                current_plan=context.current_plan,
            )
            final_importance = self.importance_scorer.score(scoring_context)
        else:
            final_importance = clamp_importance(final_importance)

        self.memory_stream.add_memory(
            node_type=NodeType.OBSERVATION,
            citations=None,
            content=content,
            now=now,
            importance=final_importance,
            embedding=embedding,
        )

        return self.memory_stream.memories[-1]

    def create_observation_from_text(
        self,
        *,
        content: str,
        now: datetime.datetime,
        context: ObservationContext,
        importance: int | None = None,
    ) -> MemoryObject:
        embedding = self.embedding_encoder.encode(
            EmbeddingEncodingContext(text=content)
        )
        return self.create_observation(
            content=content,
            now=now,
            embedding=embedding,
            context=context,
            importance=importance,
        )

    def create_reflection(
        self,
        insight: InsightWithCitation,
        *,
        now: datetime.datetime,
    ) -> MemoryObject:
        embedding = self.embedding_encoder.encode(
            EmbeddingEncodingContext(text=insight.context)
        )

        self.memory_stream.add_memory(
            node_type=NodeType.REFLECTION,
            citations=None,
            content=insight.context,
            now=now,
            importance=3,  # TODO: importance 계산 로직 필요
            embedding=embedding,
        )

        return self.memory_stream.memories[-1]
