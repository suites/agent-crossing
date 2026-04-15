from importlib import import_module
from typing import Literal, Protocol, TypeVar, cast

from typing_extensions import TypedDict

from .reaction import ReactionDecision

LANGGRAPH_GRAPH_MODULE = import_module("langgraph.graph")
GRAPH_START = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "START"))
GRAPH_END = cast(object, getattr(LANGGRAPH_GRAPH_MODULE, "END"))
TStateValue = TypeVar("TStateValue")


class AgentBrainGraphBuilder(Protocol):
    def add_node(self, node: str, action: object) -> None: ...

    def add_edge(self, start_key: object, end_key: object) -> None: ...

    def add_conditional_edges(
        self,
        source: str,
        path: object,
        path_map: dict[str, object],
    ) -> None: ...

    def compile(self) -> "AgentBrainGraphInvoker": ...


class AgentBrainGraphInvoker(Protocol):
    def invoke(self, input: "AgentBrainGraphState") -> "AgentBrainGraphState": ...


class StateGraphFactory(Protocol):
    def __call__(
        self, state_schema: type["AgentBrainGraphState"]
    ) -> AgentBrainGraphBuilder: ...


STATE_GRAPH = cast(StateGraphFactory, getattr(LANGGRAPH_GRAPH_MODULE, "StateGraph"))


class AgentBrainGraphState(TypedDict):
    input: object
    observation: object | None
    should_reflect: bool
    determine_context: object | None
    reaction_decision: ReactionDecision | None
    result: object | None


class AgentBrainStageRunner(Protocol):
    def perceive(self, input: object) -> object: ...

    def persist_observation(self, observation: object, input: object) -> bool: ...

    def run_reflection(self, input: object) -> None: ...

    def determine_context(self, observation: object, input: object) -> object: ...

    def decide_reaction(self, determine_context: object) -> ReactionDecision: ...

    def finalize_action(
        self,
        *,
        input: object,
        reaction_decision: ReactionDecision,
    ) -> object: ...


class AgentBrainGraphRunner:
    def __init__(self, *, stage_runner: object):
        self.stage_runner: object = stage_runner
        self.graph: AgentBrainGraphInvoker = self._build_graph()

    def run(self, input: object) -> object:
        final_state = self.graph.invoke(
            AgentBrainGraphState(
                input=input,
                observation=None,
                should_reflect=False,
                determine_context=None,
                reaction_decision=None,
                result=None,
            )
        )
        if final_state["result"] is None:
            raise RuntimeError("Agent brain graph did not produce a result")
        return final_state["result"]

    def _build_graph(self) -> AgentBrainGraphInvoker:
        builder = STATE_GRAPH(AgentBrainGraphState)
        # 1. 현재 상황을 인지한다. 인지할때 월드에서 현재 상황을 조회해서 주입한다.
        builder.add_node("perceive", self._perceive)
        # 2. 인지된 정보들을 observation으로 메모리에 저장 (reflection 조건 충족 시 reflection도 함께 저장)
        builder.add_node("persist_observation", self._persist_observation)
        # 2-1. reflection 조건 충족 시 reflection graph를 실행한다.
        builder.add_node("run_reflection", self._run_reflection)
        # 3. 상황판단을 한다.
        builder.add_node("determine_context", self._determine_context)
        # 4. 상황판단에 따라 반응을 결정한다.
        builder.add_node("decide_reaction", self._decide_reaction)
        # 5. 반응에 때라 구체적인 행동 및 출력을 한다.
        builder.add_node("finalize_action", self._finalize_action)

        builder.add_edge(GRAPH_START, "perceive")
        builder.add_edge("perceive", "persist_observation")
        builder.add_conditional_edges(
            "persist_observation",
            self._route_after_persist_observation,
            {
                "run_reflection": "run_reflection",
                "determine_context": "determine_context",
            },
        )
        builder.add_edge("run_reflection", "determine_context")
        builder.add_edge("determine_context", "decide_reaction")
        builder.add_edge("decide_reaction", "finalize_action")
        builder.add_edge("finalize_action", GRAPH_END)
        return builder.compile()

    def _perceive(self, state: AgentBrainGraphState) -> dict[str, object]:
        return {
            "observation": cast(AgentBrainStageRunner, self.stage_runner).perceive(
                state["input"]
            )
        }

    def _persist_observation(self, state: AgentBrainGraphState) -> dict[str, bool]:
        return {
            "should_reflect": cast(
                AgentBrainStageRunner,
                self.stage_runner,
            ).persist_observation(
                _require_state_value(state["observation"], key="observation"),
                state["input"],
            )
        }

    def _route_after_persist_observation(
        self, state: AgentBrainGraphState
    ) -> Literal["run_reflection", "determine_context"]:
        if state["should_reflect"]:
            return "run_reflection"
        return "determine_context"

    def _run_reflection(self, state: AgentBrainGraphState) -> dict[str, bool]:
        cast(AgentBrainStageRunner, self.stage_runner).run_reflection(state["input"])
        return {"should_reflect": False}

    def _determine_context(self, state: AgentBrainGraphState) -> dict[str, object]:
        return {
            "determine_context": cast(
                AgentBrainStageRunner,
                self.stage_runner,
            ).determine_context(
                _require_state_value(state["observation"], key="observation"),
                state["input"],
            )
        }

    def _decide_reaction(
        self, state: AgentBrainGraphState
    ) -> dict[str, ReactionDecision]:
        return {
            "reaction_decision": cast(
                AgentBrainStageRunner,
                self.stage_runner,
            ).decide_reaction(
                _require_state_value(
                    state["determine_context"],
                    key="determine_context",
                )
            )
        }

    def _finalize_action(self, state: AgentBrainGraphState) -> dict[str, object]:
        return {
            "result": cast(
                AgentBrainStageRunner,
                self.stage_runner,
            ).finalize_action(
                input=state["input"],
                reaction_decision=_require_state_value(
                    state["reaction_decision"],
                    key="reaction_decision",
                ),
            )
        }


def _require_state_value(value: TStateValue | None, *, key: str) -> TStateValue:
    if value is None:
        raise RuntimeError(f"Agent brain graph missing required state: {key}")
    return value
