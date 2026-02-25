import datetime
from dataclasses import dataclass

import numpy as np
from memory.memory_object import MemoryObject
from memory.memory_service import MemoryService

from .reflection_service import ReflectionService


@dataclass(frozen=True)
class PendingObservation:
    content: str
    now: datetime.datetime
    embedding: np.ndarray | None
    persona: str | None
    current_plan: str | None
    importance: int | None


class AgentBrain:
    """starter template 수준의 최소 AgentBrain 스켈레톤."""

    def __init__(
        self,
        *,
        memory_service: MemoryService,
        reflection_service: ReflectionService,
    ):
        self.memory_service: MemoryService = memory_service
        self.reflection_service: ReflectionService = reflection_service
        self._pending_observations: list[PendingObservation] = []

    def loop(self) -> None:
        """
        AgentBrain의 메인 루프.
        """
        # 1. 현재 상황을 인지한다. 인지할때 월드에서 현재 상황을 조회해서 주입한다.
        # 2. 인지된 정보들을 observation으로 저장 (reflection 조건 충족 시 reflection도 함께 저장)
        # self.create_observation(...)
        # 3. 상황판단을 한다.
        # 3-1. 상황판단을 위해 기억을 검색한다.
        # 4. 상황판단에 따라 반응을 결정한다.
        # 4-1. 관찰 결과가 단순하다면 기존 계획을 수행한다.
        # 4-2. 관찰 결과가 중요하거나 예상치 못하면 기존 계획을 멈추고 반응한다.
        # 4-2-1. 반응하기로 결정했으면 행동 계획을 재생성하고, 대화중이라면 자연어 대화도 생성한다.
        # 5. 구체적인 행동 결정 및 출력을 한다.
        # 5-1. 할 일이 정해지면 자신이 알고있는 환경트리를 탐색해서 해당 행동을 수행할 적절한 장소를 결정한다.
        # 5-2. 행동 이후에 월드 데이터에 업데이트한다.

        for observation in self._pending_observations:
            self.create_observation(
                content=observation.content,
                now=observation.now,
                embedding=observation.embedding,
                persona=observation.persona,
                current_plan=observation.current_plan,
                importance=observation.importance,
            )

        self._pending_observations.clear()
        self.reflection_service.reflect()

    def queue_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray | None = None,
        persona: str | None = None,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> None:
        self._pending_observations.append(
            PendingObservation(
                content=content,
                now=now,
                embedding=embedding,
                persona=persona,
                current_plan=current_plan,
                importance=importance,
            )
        )

    def perceive(self, *, now: datetime.datetime):
        """
        현재 상황을 인지한다.
        """
        _ = now

        # 1. 시야 범위 내의 환경 및 객체 인식
        # 2. (TODO:) 환경의 트리 구조 파악 및 자연어 변환
        pass

    def create_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        embedding: np.ndarray | None = None,
        persona: str | None = None,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> None:
        """
        메모:
        - 목적: observation 저장 + (조건 충족 시) reflection 엔트리 호출.
        - 입력/출력: 관찰 입력값들 -> (observation 1개, reflection 리스트)
        """
        created_memory: MemoryObject
        if embedding is None:
            created_memory = self.memory_service.create_observation_from_text(
                content=content,
                now=now,
                persona=persona,
                current_plan=current_plan,
                importance=importance,
            )
        else:
            created_memory = self.memory_service.create_observation(
                content=content,
                now=now,
                embedding=embedding,
                persona=persona,
                current_plan=current_plan,
                importance=importance,
            )

        self.reflection_service.record_observation_importance(
            importance=created_memory.importance
        )
