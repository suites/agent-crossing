import datetime

import numpy as np

from memory.memory_object import MemoryObject
from memory.memory_service import MemoryService

from .reflection_pipeline import ReflectionPipelineService


class AgentBrain:
    """
    최소 오케스트레이션 스켈레톤.

    현재는 관찰 저장 -> reflection trigger 확인 -> reflection 실행만 연결한다.
    실제 perceive/plan/act/react 로직은 추후 확장한다.
    """

    def __init__(
        self,
        *,
        memory_service: MemoryService,
        reflection_pipeline: ReflectionPipelineService | None = None,
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
        메모 노트:
        - 의도: 관찰 입력을 저장하고 필요 시 reflection 스캐폴딩 실행까지 연결한다.
        - 입력: 관찰 텍스트/시간/임베딩(+선택 컨텍스트)
        - 출력: (생성된 observation, 생성된 reflection들)
        - TODO: tick 루프(perceive->retrieve->plan->act->react)와 결합
        - 엣지케이스: reflection_pipeline이 없으면 observation만 반환
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

        if not self.reflection_pipeline.check_reflection_trigger():
            return observation, []

        reflections = self.reflection_pipeline.run_reflection(now=now)
        return observation, reflections
