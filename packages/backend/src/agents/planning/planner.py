import datetime
from typing import Protocol

from .models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)


class PlanGenerator(Protocol):
    def generate_day_plan(
        self,
        *,
        agent_name: str,
        age: int,
        innate_traits: list[str],
        persona_background: str,
        yesterday_date: datetime.datetime,
        yesterday_summary: str,
        today_date: datetime.datetime,
    ) -> list[DayPlanItem]: ...

    def generate_hour_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        day_plan_item: DayPlanItem,
    ) -> list[HourlyPlanItem]: ...

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_item: HourlyPlanItem,
    ) -> list[MinutePlanItem]: ...


class Planner:
    def __init__(self, plan_generator: PlanGenerator):
        """계획 생성을 위임하는 생성기 구현체."""
        self.plan_generator: PlanGenerator = plan_generator

    def generate_day_plan(
        self,
        request: DayPlanBroadStrokesRequest,
    ) -> list[DayPlanItem]:
        """일일 계획을 생성합니다."""
        return self.plan_generator.generate_day_plan(
            agent_name=request.agent_name,
            age=request.age,
            innate_traits=request.innate_traits,
            persona_background=request.persona_background,
            yesterday_date=request.yesterday_date,
            yesterday_summary=request.yesterday_summary,
            today_date=request.today_date,
        )

    def generate_hourly_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        day_plan_item: DayPlanItem,
    ) -> list[HourlyPlanItem]:
        """시간 단위 계획을 생성합니다."""
        return self.plan_generator.generate_hour_plan(
            agent_name=agent_name,
            current_time=current_time,
            day_plan_item=day_plan_item,
        )

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_item: HourlyPlanItem,
    ) -> list[MinutePlanItem]:
        """분 단위 계획을 생성합니다."""
        return self.plan_generator.generate_minute_plan(
            agent_name=agent_name,
            current_time=current_time,
            hourly_plan_item=hourly_plan_item,
        )
