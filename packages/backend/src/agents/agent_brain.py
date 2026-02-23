import datetime

import numpy as np

from memory.memory_object import MemoryObject
from memory.memory_service import MemoryService

from .reflection_pipeline import ReflectionPipeline


class AgentBrain:
    """starter template 수준의 최소 AgentBrain 스켈레톤."""

    def __init__(
        self,
        *,
        memory_service: MemoryService,
        reflection_pipeline: ReflectionPipeline | None = None,
    ):
        self.memory_service = memory_service
        self.reflection_pipeline = reflection_pipeline

    def ingest_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray,
        persona: str | None = None,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> tuple[MemoryObject, list[MemoryObject]]:
        """
        메모:
        - 목적: observation 저장 + (조건 충족 시) reflection 엔트리 호출.
        - 입력/출력: 관찰 입력값들 -> (observation 1개, reflection 리스트)
        - 다음 구현 위치: perceive/retrieve/plan/act/react 루프는 여기서 확장.
        """
        observation = self.memory_service.create_observation(
            content=content,
            now=now,
            embedding=embedding,
            persona=persona,
            current_plan=current_plan,
            importance=importance,
        )

        if self.reflection_pipeline is None:
            return observation, []

        if not self.reflection_pipeline.should_reflect():
            return observation, []

        return observation, self.reflection_pipeline.run(now=now)
