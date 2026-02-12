import datetime

import numpy as np
import pytest

from memory.memory_object import NodeType
from memory.memory_stream import MemoryStream


@pytest.fixture
def stream():
    return MemoryStream()


@pytest.fixture
def now():
    return datetime.datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def embedding():
    rng = np.random.default_rng(42)
    return rng.random(384)


@pytest.fixture
def observation_kwargs(now, embedding):
    return {
        "node_type": NodeType.OBSERVATION,
        "citations": None,
        "description": "지호가 도서관에서 책을 빌렸다.",
        "now": now,
        "importance": 7,
        "embedding": embedding,
    }


@pytest.fixture
def stream_with_memory(stream, observation_kwargs):
    stream.add_memory(**observation_kwargs)
    return stream


def test_add_memory(stream, observation_kwargs, now, embedding):
    stream.add_memory(**observation_kwargs)

    assert len(stream.memories) == 1

    memory = stream.memories[0]
    assert memory.id == 0
    assert memory.description == observation_kwargs["description"]
    assert memory.importance == observation_kwargs["importance"]
    assert memory.creation_timestamp == now
    assert np.array_equal(memory.embedding, embedding)

    stream.add_memory(
        node_type=NodeType.OBSERVATION,
        citations=None,
        description="두 번째 기억",
        now=now,
        importance=3,
        embedding=np.zeros(384),
    )
    assert len(stream.memories) == 2
    assert stream.memories[1].id == 1
