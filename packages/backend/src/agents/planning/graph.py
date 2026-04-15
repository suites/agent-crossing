import datetime
from importlib import import_module
from typing import Protocol, cast

from typing_extensions import TypedDict

from .models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)

LANGGRAPH_GRAPH_MODULE = import_module("langgraph.graph")
GRAPH_START = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "START"))
GRAPH_END = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "END"))


class PlanningGraphBuilder(Protocol):
    def add_node(self, node: str, action: object) -> None: ...

    def add_edge(self, start_key: object, end_key: object) -> None: ...

    def compile(self) -> "PlanningGraphInvoker": ...


class PlanningGraphInvoker(Protocol):
    def invoke(self, input: object) -> object: ...


class StateGraphFactory(Protocol):
    def __call__(self, state_schema: type[object]) -> PlanningGraphBuilder: ...


STATE_GRAPH = cast(StateGraphFactory, getattr(LANGGRAPH_GRAPH_MODULE, "StateGraph"))


class PlanningGraphGenerator(Protocol):
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


class DayPlanningGraphState(TypedDict):
    request: DayPlanBroadStrokesRequest
    plan_items: list[DayPlanItem]


class HourlyPlanningGraphState(TypedDict):
    agent_name: str
    current_time: datetime.datetime
    day_plan_item: DayPlanItem
    plan_items: list[HourlyPlanItem]


class MinutePlanningGraphState(TypedDict):
    agent_name: str
    current_time: datetime.datetime
    hourly_plan_item: HourlyPlanItem
    plan_items: list[MinutePlanItem]


class PlanningGraphRunner:
    def __init__(self, *, plan_generator: PlanningGraphGenerator):
        self.plan_generator: PlanningGraphGenerator = plan_generator
        self.day_plan_graph: PlanningGraphInvoker = self._build_day_plan_graph()
        self.hourly_plan_graph: PlanningGraphInvoker = self._build_hourly_plan_graph()
        self.minute_plan_graph: PlanningGraphInvoker = self._build_minute_plan_graph()

    def generate_day_plan(
        self,
        request: DayPlanBroadStrokesRequest,
    ) -> list[DayPlanItem]:
        final_state = cast(
            DayPlanningGraphState,
            self.day_plan_graph.invoke(
                DayPlanningGraphState(request=request, plan_items=[])
            ),
        )
        return final_state["plan_items"]

    def generate_hourly_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        day_plan_item: DayPlanItem,
    ) -> list[HourlyPlanItem]:
        final_state = cast(
            HourlyPlanningGraphState,
            self.hourly_plan_graph.invoke(
                HourlyPlanningGraphState(
                    agent_name=agent_name,
                    current_time=current_time,
                    day_plan_item=day_plan_item,
                    plan_items=[],
                )
            ),
        )
        return final_state["plan_items"]

    def generate_minute_plan(
        self,
        *,
        agent_name: str,
        current_time: datetime.datetime,
        hourly_plan_item: HourlyPlanItem,
    ) -> list[MinutePlanItem]:
        final_state = cast(
            MinutePlanningGraphState,
            self.minute_plan_graph.invoke(
                MinutePlanningGraphState(
                    agent_name=agent_name,
                    current_time=current_time,
                    hourly_plan_item=hourly_plan_item,
                    plan_items=[],
                )
            ),
        )
        return final_state["plan_items"]

    def _build_day_plan_graph(self) -> PlanningGraphInvoker:
        builder = STATE_GRAPH(DayPlanningGraphState)
        builder.add_node("generate_day_plan", self._generate_day_plan)
        builder.add_edge(GRAPH_START, "generate_day_plan")
        builder.add_edge("generate_day_plan", GRAPH_END)
        return builder.compile()

    def _build_hourly_plan_graph(self) -> PlanningGraphInvoker:
        builder = STATE_GRAPH(HourlyPlanningGraphState)
        builder.add_node("generate_hourly_plan", self._generate_hourly_plan)
        builder.add_edge(GRAPH_START, "generate_hourly_plan")
        builder.add_edge("generate_hourly_plan", GRAPH_END)
        return builder.compile()

    def _build_minute_plan_graph(self) -> PlanningGraphInvoker:
        builder = STATE_GRAPH(MinutePlanningGraphState)
        builder.add_node("generate_minute_plan", self._generate_minute_plan)
        builder.add_edge(GRAPH_START, "generate_minute_plan")
        builder.add_edge("generate_minute_plan", GRAPH_END)
        return builder.compile()

    def _generate_day_plan(
        self,
        state: DayPlanningGraphState,
    ) -> dict[str, list[DayPlanItem]]:
        request = state["request"]
        return {
            "plan_items": self.plan_generator.generate_day_plan(
                agent_name=request.agent_name,
                age=request.age,
                innate_traits=request.innate_traits,
                persona_background=request.persona_background,
                yesterday_date=request.yesterday_date,
                yesterday_summary=request.yesterday_summary,
                today_date=request.today_date,
            )
        }

    def _generate_hourly_plan(
        self,
        state: HourlyPlanningGraphState,
    ) -> dict[str, list[HourlyPlanItem]]:
        return {
            "plan_items": self.plan_generator.generate_hour_plan(
                agent_name=state["agent_name"],
                current_time=state["current_time"],
                day_plan_item=state["day_plan_item"],
            )
        }

    def _generate_minute_plan(
        self,
        state: MinutePlanningGraphState,
    ) -> dict[str, list[MinutePlanItem]]:
        return {
            "plan_items": self.plan_generator.generate_minute_plan(
                agent_name=state["agent_name"],
                current_time=state["current_time"],
                hourly_plan_item=state["hourly_plan_item"],
            )
        }
