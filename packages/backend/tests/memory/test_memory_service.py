import datetime

import numpy as np
from agents.memory.memory_service import MemoryService
from agents.memory.memory_stream import MemoryStream
from llm import ImportanceScoringContext
from llm.embedding_encoder import EmbeddingEncodingContext
from settings import EMBEDDING_DIMENSION


class StubScorer:
    def __init__(self, score_value: int):
        self.score_value: int = score_value
        self.last_context: ImportanceScoringContext | None = None

    def score(self, context: ImportanceScoringContext) -> int:
        self.last_context = context
        return self.score_value


class StubEmbeddingEncoder:
    def encode(self, _context: EmbeddingEncodingContext) -> np.ndarray:
        return np.zeros(EMBEDDING_DIMENSION)


def test_create_observation_uses_scorer_when_importance_missing() -> None:
    stream = MemoryStream()
    scorer = StubScorer(score_value=9)
    service = MemoryService(
        memory_stream=stream,
        importance_scorer=scorer,
        embedding_encoder=StubEmbeddingEncoder(),
    )

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(EMBEDDING_DIMENSION)

    memory = service.create_observation(
        content="카페 계약이 성사됐다.",
        now=now,
        embedding=embedding,
        context=ObservationContext(
            agent_name="Sujin Lee",
            identity_stable_set=["Sujin values reliable service."],
            current_plan="오후에 계약서 서명",
        ),
    )

    assert memory.importance == 9
    assert scorer.last_context is not None
    assert scorer.last_context.observation == "카페 계약이 성사됐다."


def test_create_observation_clamps_explicit_importance() -> None:
    stream = MemoryStream()
    scorer = StubScorer(score_value=5)
    service = MemoryService(
        memory_stream=stream,
        importance_scorer=scorer,
        embedding_encoder=StubEmbeddingEncoder(),
    )

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(EMBEDDING_DIMENSION)

    memory = service.create_observation(
        content="반복적인 일상 기록",
        now=now,
        embedding=embedding,
        context=ObservationContext(agent_name="Sujin Lee", identity_stable_set=[]),
        importance=42,
    )

    assert memory.importance == 10


def test_create_observation_uses_scorer_value() -> None:
    stream = MemoryStream()
    scorer = StubScorer(score_value=3)
    service = MemoryService(
        memory_stream=stream,
        importance_scorer=scorer,
        embedding_encoder=StubEmbeddingEncoder(),
    )

    now = datetime.datetime(2026, 2, 13, 12, 0, 0)
    embedding = np.zeros(EMBEDDING_DIMENSION)

    memory = service.create_observation(
        content="마을 광장을 산책했다.",
        now=now,
        embedding=embedding,
        context=ObservationContext(agent_name="Jiho Park", identity_stable_set=[]),
    )

    assert memory.importance == 3


# class StubReflectionService:
#     def __init__(self):
#         self.recorded_importance: list[int] = []

#     def record_observation_importance(self, importance: int) -> None:
#         self.recorded_importance.append(importance)


# def test_create_observation_records_importance_to_reflection_service() -> None:
#     stream = MemoryStream()
#     reflection_service = StubReflectionService()
#     service = MemoryService(memory_stream=stream, reflection_service=reflection_service)

#     now = datetime.datetime(2026, 2, 13, 12, 0, 0)
#     embedding = np.zeros(EMBEDDING_DIMENSION)

#     _ = service.create_observation(
#         content="중요한 사건 발생",
#         now=now,
#         embedding=embedding,
#         importance=8,
#     )

#     assert reflection_service.recorded_importance == [8]
