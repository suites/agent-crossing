import datetime

import numpy as np
from memory.memory_service import MemoryService

from .reflection_service import ReflectionService


class AgentBrain:
    """starter template 수준의 최소 AgentBrain 스켈레톤."""

    def __init__(
        self,
        *,
        memory_service: MemoryService,
        reflection_service: ReflectionService,
    ):
        self.memory_service = memory_service
        self.reflection_service = reflection_service

    def ingest_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray,
        persona: str | None = None,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> None:
        """
        메모:
        - 목적: observation 저장 + (조건 충족 시) reflection 엔트리 호출.
        - 입력/출력: 관찰 입력값들 -> (observation 1개, reflection 리스트)
        """
        self.memory_service.create_observation(
            content=content,
            now=now,
            embedding=embedding,
            persona=persona,
            current_plan=current_plan,
            importance=importance,
        )

        self.reflection_service.reflect()
