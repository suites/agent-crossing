from typing import List

from .memory_object import MemoryObject


class MemoryService:
    def __init__(self, memory_stream: List[MemoryObject]):
        self.memory_stream = memory_stream
