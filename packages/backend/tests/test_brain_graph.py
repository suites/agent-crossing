import datetime
from typing import Literal, cast

import numpy as np
from agents.agent import AgentProfile, ExtendedPersona, FixedPersona
from agents.agent_brain import (
    ActionLoopInput,
    ActionLoopResult,
    DetermineContext,
    Observation,
)
from agents.brain_graph import AgentBrainGraphRunner
from agents.reaction import ReactionDecision, ReactionDecisionTrace


class StubStageRunner:
    def __init__(self, *, should_reflect: bool):
        self.should_reflect_value: bool = should_reflect
        self.calls: list[str] = []

    def perceive(self, input: ActionLoopInput) -> Observation:
        self.calls.append("perceive")
        return Observation(
            content="observation",
            now=input.current_time,
            embedding=np.zeros(1, dtype=np.float32),
            agent_name="Jiho",
            current_plan=None,
            importance=None,
        )

    def persist_observation(
        self, observation: Observation, input: ActionLoopInput
    ) -> bool:
        _ = observation
        _ = input
        self.calls.append("persist_observation")
        return self.should_reflect_value

    def run_reflection(self, input: ActionLoopInput) -> None:
        _ = input
        self.calls.append("run_reflection")

    def determine_context(
        self, observation: Observation, input: ActionLoopInput
    ) -> DetermineContext:
        _ = input
        self.calls.append("determine_context")
        return DetermineContext(
            observation=observation,
            retrieved_memories=[],
            dialogue_history=[],
            profile=input.profile,
            language="ko",
        )

    def decide_reaction(self, determine_context: DetermineContext) -> ReactionDecision:
        _ = determine_context
        self.calls.append("decide_reaction")
        return ReactionDecision(
            should_react=False,
            reaction="",
            reason="skip",
            trace=ReactionDecisionTrace(raw_response="", parse_success=True),
        )

    def finalize_action(
        self,
        *,
        input: ActionLoopInput,
        reaction_decision: ReactionDecision,
    ) -> ActionLoopResult:
        _ = input
        self.calls.append("finalize_action")
        return ActionLoopResult(
            current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
            talk=None,
            silent_reason=reaction_decision.reason,
            reaction_trace=reaction_decision.trace,
        )


class StubProfile:
    def __init__(self) -> None:
        self.fixed: FixedPersona = FixedPersona(identity_stable_set=["kind"])
        self.extended: ExtendedPersona = ExtendedPersona(
            lifestyle_and_routine=[],
            current_plan_context=[],
        )


def _input() -> ActionLoopInput:
    return ActionLoopInput(
        current_time=datetime.datetime(2026, 3, 3, 12, 0, 0),
        dialogue_history=[],
        profile=cast(AgentProfile, cast(object, StubProfile())),
        language=cast(Literal["ko", "en"], "ko"),
    )


def test_brain_graph_skips_reflection_when_not_needed() -> None:
    runner = StubStageRunner(should_reflect=False)
    graph = AgentBrainGraphRunner(stage_runner=runner)

    result = cast(ActionLoopResult, graph.run(_input()))

    assert result.silent_reason == "skip"
    assert runner.calls == [
        "perceive",
        "persist_observation",
        "determine_context",
        "decide_reaction",
        "finalize_action",
    ]


def test_brain_graph_runs_reflection_before_determine_context() -> None:
    runner = StubStageRunner(should_reflect=True)
    graph = AgentBrainGraphRunner(stage_runner=runner)

    _ = graph.run(_input())

    assert runner.calls == [
        "perceive",
        "persist_observation",
        "run_reflection",
        "determine_context",
        "decide_reaction",
        "finalize_action",
    ]
