import datetime

import numpy as np

from agents.reflection_pipeline import ReflectionConfig, ReflectionPipelineService
from memory.memory_object import MemoryObject, NodeType
from memory.memory_stream import MemoryStream


class StubQuestionGenerator:
    def __init__(self):
        self.called_with_count: int | None = None

    def generate(self, memories: list[MemoryObject], count: int) -> list[str]:
        self.called_with_count = count
        return ["Q1", "Q2", "Q3"]


class StubRetriever:
    def retrieve(
        self,
        *,
        question: str,
        memories: list[MemoryObject],
        top_k: int,
    ) -> list[MemoryObject]:
        return memories[:top_k]


class StubInsightGenerator:
    def __init__(self):
        self.called_with_count: int | None = None

    def generate(
        self,
        *,
        questions: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
        count: int,
    ) -> list[str]:
        self.called_with_count = count
        return ["I1", "I2"]


class StubCitationLinker:
    def link(
        self,
        *,
        insights: list[str],
        retrieved_by_question: dict[str, list[MemoryObject]],
    ) -> list[list[int]]:
        return [[0], [0, 1]]


class StubRollover:
    def rollover(self, *, accumulated_importance_before_run: int) -> int:
        # 초과분 이월 예시(스켈레톤 계약 확인용)
        return max(0, accumulated_importance_before_run - 150)


def _seed_observation(stream: MemoryStream, now: datetime.datetime, importance: int = 7) -> None:
    stream.add_memory(
        node_type=NodeType.OBSERVATION,
        citations=None,
        content="seed",
        now=now,
        importance=importance,
        embedding=np.array([1.0, 0.0, 0.0]),
    )


def test_check_reflection_trigger_threshold_contract() -> None:
    stream = MemoryStream()
    service = ReflectionPipelineService(
        memory_stream=stream,
        config=ReflectionConfig(threshold=15),
    )

    service.record_observation_importance(10)
    assert service.check_reflection_trigger() is False

    service.record_observation_importance(5)
    assert service.check_reflection_trigger() is True


def test_run_reflection_skeleton_creates_reflection_and_applies_rollover() -> None:
    stream = MemoryStream()
    now = datetime.datetime(2026, 2, 23, 18, 0, 0)
    _seed_observation(stream, now)
    _seed_observation(stream, now)

    qg = StubQuestionGenerator()
    ig = StubInsightGenerator()

    service = ReflectionPipelineService(
        memory_stream=stream,
        config=ReflectionConfig(threshold=150, question_count=3, insight_count=5),
        question_generator=qg,
        retriever=StubRetriever(),
        insight_generator=ig,
        citation_linker=StubCitationLinker(),
        rollover_policy=StubRollover(),
    )
    service.accumulated_importance = 170

    reflections = service.run_reflection(now=now)

    assert qg.called_with_count == 3
    assert ig.called_with_count == 5
    assert len(reflections) == 2
    assert all(m.node_type == NodeType.REFLECTION for m in reflections)
    assert reflections[0].citations == [0]
    assert reflections[1].citations == [0, 1]
    assert service.accumulated_importance == 20
