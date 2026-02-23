import datetime

import numpy as np
from llm import ImportanceScorer, ImportanceScoringContext, clamp_importance

from .memory_object import MemoryObject, NodeType
from .memory_stream import MemoryStream


class MemoryService:
    def __init__(
        self,
        memory_stream: MemoryStream,
        importance_scorer: ImportanceScorer | None = None,
    ):
        self.memory_stream: MemoryStream = memory_stream
        self.importance_scorer: ImportanceScorer | None = importance_scorer

    def create_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray,
        persona: str | None = None,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> MemoryObject:
        final_importance = importance
        if final_importance is None:
            if self.importance_scorer is None:
                final_importance = 3
            else:
                context = ImportanceScoringContext(
                    observation=content,
                    persona=persona,
                    current_plan=current_plan,
                )
                final_importance = self.importance_scorer.score(context)
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
