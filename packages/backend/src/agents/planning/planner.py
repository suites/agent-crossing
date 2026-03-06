from typing import Protocol

from .models import DayPlanBroadStrokesRequest, DayPlanItem


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

    def generate_hourly_plan(self):
        """시간 단위 계획을 생성합니다."""

    def generate_minute_plan(self):
        """분 단위 계획을 생성합니다."""
