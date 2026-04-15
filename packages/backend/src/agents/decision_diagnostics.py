from dataclasses import dataclass

from agents.reaction import ReactionDecision


@dataclass(frozen=True)
class ActionDiagnostics:
    thought: str
    model_thought: str
    self_critique: str
    decision_reason: str
    action_summary: str
    decision_process: dict[str, object]


def build_action_diagnostics(
    *,
    reaction_decision: ReactionDecision,
    speak_decision: bool,
    action_intent: str,
    silent_reason: str,
) -> ActionDiagnostics:
    return ActionDiagnostics(
        thought=(reaction_decision.critique or reaction_decision.reason),
        model_thought=reaction_decision.thought,
        self_critique=reaction_decision.critique,
        decision_reason=reaction_decision.reason,
        action_summary=(
            f"speak_decision={speak_decision}, "
            f"action_intent={action_intent}, "
            f"should_react={reaction_decision.should_react}, "
            f"reason={reaction_decision.reason or 'n/a'}"
        ),
        decision_process={
            "llm_decision": {
                "should_react": reaction_decision.should_react,
                "reason": reaction_decision.reason,
                "model_thought": reaction_decision.thought,
                "self_critique": reaction_decision.critique,
                "candidate_reaction": reaction_decision.reaction,
            },
            "action": {
                "speak_decision": speak_decision,
                "action_intent": action_intent,
                "silent_reason": silent_reason,
            },
        },
    )
