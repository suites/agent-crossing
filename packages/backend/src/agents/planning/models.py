import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class DayPlanItem:
    start_time: datetime.datetime
    duration_minutes: int
    location: str
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
    start_time: datetime.datetime
    duration_minutes: int
    location: str
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
    start_time: datetime.datetime
    duration_minutes: int
    location: str
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
    items: list[DayPlanItem]

    def __post_init__(self) -> None:
        if len(self.items) < 5 or len(self.items) > 8:
            raise ValueError("day plan must contain between 5 and 8 items")


@dataclass(frozen=True)
class HourlyPlan:
    items: list[HourlyPlanItem]


@dataclass(frozen=True)
class MinutePlan:
    items: list[MinutePlanItem]


@dataclass(frozen=True)
class DayPlanBroadStrokesRequest:
    agent_name: str
    age: int
    innate_traits: list[str]
    persona_background: str
    yesterday_date_text: str
    yesterday_summary: str
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
    items: list[str]

    def __post_init__(self) -> None:
        filtered_items = [item for item in self.items if item.strip()]
        if len(filtered_items) < 5 or len(filtered_items) > 8:
            raise ValueError("broad strokes must contain between 5 and 8 items")
