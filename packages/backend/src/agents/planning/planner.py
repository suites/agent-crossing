from typing import Protocol

from .models import DayPlanBroadStrokes, DayPlanBroadStrokesRequest


class DayPlanGenerator(Protocol):
    def generate_day_plan_broad_strokes(
        self,
        *,
        agent_name: str,
        age: int,
        innate_traits: list[str],
        persona_background: str,
        yesterday_date_text: str,
        yesterday_summary: str,
        today_date_text: str,
    ) -> list[str]: ...


class Planner:
    def __init__(self, day_plan_generator: DayPlanGenerator):
        self.day_plan_generator: DayPlanGenerator = day_plan_generator

    def generate_day_plan_broad_strokes(
        self,
        request: DayPlanBroadStrokesRequest,
    ) -> DayPlanBroadStrokes:
        """Generate validated broad-strokes plan using the configured LLM generator."""
        strokes = self.day_plan_generator.generate_day_plan_broad_strokes(
            agent_name=request.agent_name,
            age=request.age,
            innate_traits=request.innate_traits,
            persona_background=request.persona_background,
            yesterday_date_text=request.yesterday_date_text,
            yesterday_summary=request.yesterday_summary,
            today_date_text=request.today_date_text,
        )
        return DayPlanBroadStrokes(items=strokes)
