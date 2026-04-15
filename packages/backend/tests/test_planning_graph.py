import datetime

from agents.planning.graph import PlanningGraphRunner
from agents.planning.models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)
from agents.planning.planner import Planner


class StubPlanGenerator:
    def __init__(self) -> None:
        self.calls: list[str] = []

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
    ) -> list[DayPlanItem]:
        self.calls.append(f"day:{agent_name}:{age}:{innate_traits[0]}")
        _ = persona_background, yesterday_date, yesterday_summary, today_date
        return [
            DayPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                end_time=datetime.datetime(2026, 2, 13, 9, 0, 0),
                location="Town > Home > Desk",
                action_content="Plan the morning composition session.",
            )
        ]

    def generate_hour_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        day_plan_item: DayPlanItem,
    ) -> list[HourlyPlanItem]:
        self.calls.append(f"hour:{agent_name}:{current_time.isoformat()}")
        _ = day_plan_item
        return [
            HourlyPlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                end_time=datetime.datetime(2026, 2, 13, 9, 0, 0),
                location="Town > Home > Desk",
                action_content="Draft composition motifs.",
            )
        ]

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_item: HourlyPlanItem,
    ) -> list[MinutePlanItem]:
        self.calls.append(f"minute:{agent_name}:{current_time.isoformat()}")
        _ = hourly_plan_item
        return [
            MinutePlanItem(
                start_time=datetime.datetime(2026, 2, 13, 8, 0, 0),
                end_time=datetime.datetime(2026, 2, 13, 8, 10, 0),
                location="Town > Home > Desk",
                action_content="Sketch the first phrase.",
            )
        ]


def _day_plan_request() -> DayPlanBroadStrokesRequest:
    return DayPlanBroadStrokesRequest(
        agent_name="Eddy Lin",
        age=19,
        innate_traits=["friendly", "outgoing"],
        persona_background="Music theory student focusing on composition.",
        yesterday_date=datetime.datetime(2026, 2, 12),
        yesterday_summary="Studied harmony and practiced composition in the evening.",
        today_date=datetime.datetime(2026, 2, 13),
    )


def test_planning_graph_runner_delegates_day_hour_and_minute_generation() -> None:
    generator = StubPlanGenerator()
    graph = PlanningGraphRunner(plan_generator=generator)

    day_items = graph.generate_day_plan(_day_plan_request())
    hourly_items = graph.generate_hourly_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 8, 30, 0),
        day_plan_item=day_items[0],
    )
    minute_items = graph.generate_minute_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 8, 10, 0),
        hourly_plan_item=hourly_items[0],
    )

    assert len(day_items) == 1
    assert len(hourly_items) == 1
    assert len(minute_items) == 1
    assert generator.calls == [
        "day:Eddy Lin:19:friendly",
        "hour:Eddy Lin:2026-02-13T08:30:00",
        "minute:Eddy Lin:2026-02-13T08:10:00",
    ]


def test_planner_uses_planning_graph_for_existing_entrypoints() -> None:
    generator = StubPlanGenerator()
    planner = Planner(generator)

    day_items = planner.generate_day_plan(_day_plan_request())
    hourly_items = planner.generate_hourly_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 8, 30, 0),
        day_plan_item=day_items[0],
    )
    minute_items = planner.generate_minute_plan(
        agent_name="Eddy Lin",
        current_time=datetime.datetime(2026, 2, 13, 8, 10, 0),
        hourly_plan_item=hourly_items[0],
    )

    assert len(day_items) == 1
    assert len(hourly_items) == 1
    assert len(minute_items) == 1
    assert generator.calls == [
        "day:Eddy Lin:19:friendly",
        "hour:Eddy Lin:2026-02-13T08:30:00",
        "minute:Eddy Lin:2026-02-13T08:10:00",
    ]
