import datetime

import numpy as np

from llm import ImportanceScoringContext
from memory.memory_service import MemoryService
from memory.memory_stream import MemoryStream


class StubScorer:
    def __init__(self, score_value: int):
        self.score_value: int = score_value
        self.last_context: ImportanceScoringContext | None = None

    def score(self, context: ImportanceScoringContext) -> int:
        self.last_context = context
        return self.score_value


def test_create_observation_uses_scorer_when_importance_missing() -> None:
    stream = MemoryStream()
    scorer = StubScorer(score_value=9)
    service = MemoryService(memory_stream=stream, importance_scorer=scorer)

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(384)

    memory = service.create_observation(
        content="카페 계약이 성사됐다.",
        now=now,
        embedding=embedding,
        persona="수진은 사업 확장을 목표로 한다.",
        current_plan="오후에 계약서 서명",
    )

    assert memory.importance == 9
    assert scorer.last_context is not None
    assert scorer.last_context.observation == "카페 계약이 성사됐다."


def test_create_observation_clamps_explicit_importance() -> None:
    stream = MemoryStream()
    service = MemoryService(memory_stream=stream, importance_scorer=None)

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(384)

    memory = service.create_observation(
        content="반복적인 일상 기록",
        now=now,
        embedding=embedding,
        importance=42,
    )

    assert memory.importance == 10


def test_create_observation_uses_default_fallback_without_scorer() -> None:
    stream = MemoryStream()
    service = MemoryService(memory_stream=stream)

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(384)

    memory = service.create_observation(
        content="마을 광장을 산책했다.",
        now=now,
        embedding=embedding,
    )

    assert memory.importance == 3


class StubReflectionPipeline:
    def __init__(self):
        self.recorded: list[int] = []

    def record_observation_importance(self, importance: int) -> None:
        self.recorded.append(importance)


def test_create_observation_records_importance_to_reflection_pipeline() -> None:
    stream = MemoryStream()
    pipeline = StubReflectionPipeline()
    service = MemoryService(memory_stream=stream, reflection_pipeline=pipeline)

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(384)

    _ = service.create_observation(
        content="중요한 사건 발생",
        now=now,
        embedding=embedding,
        importance=8,
    )

    assert pipeline.recorded == [8]
