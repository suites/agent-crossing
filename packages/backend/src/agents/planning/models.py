import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class DayPlanItem:
    """하루 계획의 단일 실행 항목."""

    """계획 시작 시각."""
    start_time: datetime.datetime

    """행동 지속 시간(분 단위)."""
    duration_minutes: int

    """행동이 수행되는 위치."""
    location: str

    """수행할 행동 내용."""
    action_content: str

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be greater than 0")
        if not self.location.strip():
            raise ValueError("location must not be blank")
        if not self.action_content.strip():
            raise ValueError("action_content must not be blank")


@dataclass(frozen=True)
class HourlyPlanItem:
    """시간 단위 계획의 단일 실행 항목."""

    """시간 단위 계획 시작 시각."""
    start_time: datetime.datetime

    """시간 단위 행동 지속 시간(분 단위)."""
    duration_minutes: int

    """시간 단위 행동이 수행되는 위치."""
    location: str

    """시간 단위로 수행할 행동 내용."""
    action_content: str

    def __post_init__(self) -> None:
        if self.duration_minutes <= 0:
            raise ValueError("duration_minutes must be greater than 0")
        if not self.location.strip():
            raise ValueError("location must not be blank")
        if not self.action_content.strip():
            raise ValueError("action_content must not be blank")


@dataclass(frozen=True)
class MinutePlanItem:
    """분 단위 계획의 단일 실행 항목."""

    """분 단위 계획 시작 시각."""
    start_time: datetime.datetime

    """분 단위 행동 지속 시간(5~15분)."""
    duration_minutes: int

    """분 단위 행동이 수행되는 위치."""
    location: str

    """즉시 실행할 분 단위 행동 내용."""
    action_content: str

    def __post_init__(self) -> None:
        if self.duration_minutes < 5 or self.duration_minutes > 15:
            raise ValueError("minute plan duration_minutes must be in range [5, 15]")
        if not self.location.strip():
            raise ValueError("location must not be blank")
        if not self.action_content.strip():
            raise ValueError("action_content must not be blank")


@dataclass(frozen=True)
class DayPlan:
    """하루 계획 전체를 나타내는 모델."""

    """하루 계획을 구성하는 항목 목록(5~8개)."""
    items: list[DayPlanItem]

    def __post_init__(self) -> None:
        if len(self.items) < 5 or len(self.items) > 8:
            raise ValueError("day plan must contain between 5 and 8 items")


@dataclass(frozen=True)
class HourlyPlan:
    """시간 단위 계획 전체를 나타내는 모델."""

    """시간 단위 계획 항목 목록."""
    items: list[HourlyPlanItem]


@dataclass(frozen=True)
class MinutePlan:
    """분 단위 실행 계획 전체를 나타내는 모델."""

    """분 단위 실행 항목 목록."""
    items: list[MinutePlanItem]


@dataclass(frozen=True)
class DayPlanBroadStrokesRequest:
    """하루 거시 계획 생성을 위한 입력 요청."""

    """계획을 생성할 에이전트 이름."""
    agent_name: str

    """에이전트 나이."""
    age: int

    """에이전트의 선천적 성향 목록."""
    innate_traits: list[str]

    """에이전트의 배경 설정 요약."""
    persona_background: str

    """어제 날짜를 나타내는 텍스트."""
    yesterday_date_text: str

    """어제 활동 요약."""
    yesterday_summary: str

    """오늘 날짜를 나타내는 텍스트."""
    today_date_text: str

    def __post_init__(self) -> None:
        if not self.agent_name.strip():
            raise ValueError("agent_name must not be blank")
        if self.age <= 0:
            raise ValueError("age must be greater than 0")
        if not any(trait.strip() for trait in self.innate_traits):
            raise ValueError("innate_traits must contain at least one non-empty trait")
        if not self.persona_background.strip():
            raise ValueError("persona_background must not be blank")
        if not self.yesterday_date_text.strip():
            raise ValueError("yesterday_date_text must not be blank")
        if not self.yesterday_summary.strip():
            raise ValueError("yesterday_summary must not be blank")
        if not self.today_date_text.strip():
            raise ValueError("today_date_text must not be blank")


@dataclass(frozen=True)
class DayPlanBroadStrokes:
    """하루 거시 계획 결과를 표현하는 모델."""

    """하루 계획의 거시적 스트로크 목록(5~8개)."""
    items: list[str]

    def __post_init__(self) -> None:
        filtered_items = [item for item in self.items if item.strip()]
        if len(filtered_items) < 5 or len(filtered_items) > 8:
            raise ValueError("broad strokes must contain between 5 and 8 items")
