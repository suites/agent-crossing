import datetime
from typing import Protocol

from .models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)


class DayPlanGenerator(Protocol):
    def generate_day_plan(
        self,
        *,
        agent_name: str,
        age: int,
        innate_traits: list[str],
        persona_background: str,
        yesterday_date_text: str,
        yesterday_summary: str,
        today_date_text: str,
    ) -> list[DayPlanItem]: ...

    def generate_hour_plan(
        self,
        *,
        agent_name: str,
        today_date_text: str,
        day_plan_items: list[DayPlanItem],
    ) -> list[HourlyPlanItem]: ...

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_items: list[HourlyPlanItem],
    ) -> list[MinutePlanItem]: ...


class Planner:
    def __init__(self, day_plan_generator: DayPlanGenerator):
        """일일 계획 생성을 위임하는 생성기 구현체."""
        self.day_plan_generator: DayPlanGenerator = day_plan_generator

    def generate_day_plan(
        self,
        request: DayPlanBroadStrokesRequest,
    ) -> list[DayPlanItem]:
        """일일 계획을 생성합니다."""
        return self.day_plan_generator.generate_day_plan(
            agent_name=request.agent_name,
            age=request.age,
            innate_traits=request.innate_traits,
            persona_background=request.persona_background,
            yesterday_date_text=request.yesterday_date_text,
            yesterday_summary=request.yesterday_summary,
            today_date_text=request.today_date_text,
        )

    def generate_hourly_plan(
        self,
        *,
        agent_name: str,
        today_date_text: str,
        day_plan_items: list[DayPlanItem],
    ) -> list[HourlyPlanItem]:
        """시간 단위 계획을 생성합니다."""
        return self.day_plan_generator.generate_hour_plan(
            agent_name=agent_name,
            today_date_text=today_date_text,
            day_plan_items=day_plan_items,
        )

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_items: list[HourlyPlanItem],
    ) -> list[MinutePlanItem]:
        """분 단위 계획을 생성합니다."""
        return self.day_plan_generator.generate_minute_plan(
            agent_name=agent_name,
            current_time=current_time,
            hourly_plan_items=hourly_plan_items,
        )
