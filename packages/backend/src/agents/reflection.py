from dataclasses import dataclass


@dataclass(frozen=True)
class ReflectionConfig:
    threshold: int = 150


class Reflection:
    def __init__(self, config: ReflectionConfig | None = None):
        self.config = config or ReflectionConfig()
        self._accumulated_importance = 0

    @property
    def accumulated_importance(self) -> int:
        """
        현재까지 누적된 중요도를 반환한다.
        """
        return self._accumulated_importance

    def record_observation_importance(self, importance: int) -> None:
        """
        observation 중요도를 누적하여 reflection 실행 조건을 계산.
        """
        self._accumulated_importance += importance

    def clear_importance(self) -> None:
        """
        누적 중요도를 초기화한다.
        """
        self._accumulated_importance = 0

    def should_reflect(self) -> bool:
        """
        현재 누적 중요도가 임계치 이상인지 판단한다.
        - 입력/출력: 없음 -> bool
        """
        return self._accumulated_importance >= self.config.threshold
