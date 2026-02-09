import datetime

import numpy as np
from memory.memory_stream import MemoryStream


def test_add_memory():
    stream = MemoryStream()
    description = "지호가 도서관에서 책을 빌렸다."
    now = datetime.datetime.now()
    importance = 7
    embedding = np.random.rand(384)

    stream.add_memory(
        description=description, now=now, importance=importance, embedding=embedding
    )

    assert len(stream.memories) == 1

    memory = stream.memories[0]
    assert memory.id == 0
    assert memory.description == description
    assert memory.importance == importance
    assert memory.creation_timestamp == now
    assert np.array_equal(memory.embedding, embedding)

    stream.add_memory("두 번째 기억", now, 3, np.zeros(384))
    assert len(stream.memories) == 2
    assert stream.memories[1].id == 1
