import datetime
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import numpy as np


class NodeType(Enum):
    OBSERVATION = "OBSERVATION"
    """ 관찰된 사실 """
    REFLECTION = "REFLECTION"
    """ 성찰한 내용 """
    PLAN = "PLAN"
    """ 계획 """


@dataclass
class MemoryObject:
    """
    메모리 객체.
    Leaf node는 항상 OBSERVATION이다.
    """

    id: int
    """고유 식별자"""

    node_type: NodeType
    """노드 타입(OBSERVATION, REFLECTION, PLAN)"""

    citations: Optional[List[int]]
    """인용된 메모리 ID 목록, OBSERVATION은 항상 None"""

    content: str
    """메모리 내용 (자연어)"""

    created_at: datetime.datetime
    """생성 시간"""

    last_accessed_at: datetime.datetime
    """마지막으로 검색된 시간"""

    importance: int
    """중요도 (1~10)"""

    embedding: np.ndarray
    """content의 임베딩 벡터 (Relevance 계산용)"""

    def update_last_accessed(self, timestamp: datetime.datetime):
        self.last_accessed_at = timestamp
