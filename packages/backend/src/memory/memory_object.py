import datetime
from dataclasses import dataclass

import numpy as np


@dataclass
class MemoryObject:
    """메모리 객체"""

    id: int
    """고유 식별자"""

    description: str
    """메모리 내용 (자연어)"""

    creation_timestamp: datetime.datetime
    """생성 시간"""

    last_accessed: datetime.datetime
    """마지막으로 검색된 시간"""

    importance: int
    """중요도 (1~10)"""

    embedding: np.ndarray
    """임베딩 벡터 (Relevance 계산용)"""

    def update_last_accessed(self, timestamp: datetime.datetime):
        self.last_accessed = timestamp
