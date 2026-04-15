import datetime

from .graph import PlanningCompletionClient, PlanningGraphRunner
from .models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)


class Planner:
    def __init__(self, planning_client: PlanningCompletionClient):
        """계획 생성을 위임하는 생성기 구현체."""
        self.planning_client: PlanningCompletionClient = planning_client
        self.planning_graph: PlanningGraphRunner = PlanningGraphRunner(
            planning_client=planning_client
        )

    def generate_day_plan(
        self,
        request: DayPlanBroadStrokesRequest,
    ) -> list[DayPlanItem]:
        """일일 계획을 생성합니다."""
        return self.planning_graph.generate_day_plan(request)

    def generate_hourly_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        day_plan_item: DayPlanItem,
    ) -> list[HourlyPlanItem]:
        """시간 단위 계획을 생성합니다."""
        return self.planning_graph.generate_hourly_plan(
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
        return self.planning_graph.generate_minute_plan(
            agent_name=agent_name,
            current_time=current_time,
            hourly_plan_item=hourly_plan_item,
        )
