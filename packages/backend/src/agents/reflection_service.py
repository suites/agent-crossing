from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import datetime

    from memory.memory_object import MemoryObject


@dataclass(frozen=True)
class ReflectionConfig:
    threshold: int = 150


class ReflectionService:
    """아주 최소한의 reflection 시작점(스캐폴딩 전용)."""

    def __init__(self, config: ReflectionConfig | None = None):
        self.config = config or ReflectionConfig()
        self.accumulated_importance = 0

    def record_observation_importance(self, importance: int) -> None:
        """
        메모:
        - 목적: observation 중요도를 누적하여 reflection 실행 조건을 계산.
        - 입력/출력: importance(int) -> None (내부 누적값 변경)
        - 다음 구현 위치: 음수 방지/세션별 분리 저장 정책은 여기서 추가.
        """
        self.accumulated_importance += importance

    def should_reflect(self) -> bool:
        """
        메모:
        - 목적: 현재 누적 중요도가 임계치 이상인지 최소 판단.
        - 입력/출력: 없음 -> bool
        - 다음 구현 위치: 시간창 기반 조건, 쿨다운 조건을 여기서 확장.
        """
        return self.accumulated_importance >= self.config.threshold

    def run(self, *, now: "datetime.datetime") -> list["MemoryObject"]:
        """
        메모:
        - 목적: reflection 실행 엔트리 포인트만 제공(실제 생성 로직은 미구현).
        - 입력/출력: now(datetime) -> list[MemoryObject] (현재는 빈 리스트)
        - 다음 구현 위치: 질문 생성/검색/인사이트 생성 파이프라인을 단계적으로 구현.
        """
        _ = now
        # starter template: 사용자가 직접 reflection 생성 로직을 채우도록 비워둔다.
        self.accumulated_importance = 0
        return []
