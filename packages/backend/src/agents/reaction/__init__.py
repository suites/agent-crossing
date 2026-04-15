from .contracts import (
    GenerateClient,
    ReactionDecision,
    ReactionDecisionInput,
    ReactionDecisionTrace,
    ReactionIntent,
    ReactionUtterance,
)

__all__ = [
    "GenerateClient",
    "ReactionDecision",
    "ReactionDecisionInput",
    "ReactionDecisionTrace",
    "ReactionIntent",
    "ReactionUtterance",
]


def __getattr__(name: str) -> object:
    if name == "ReactionGraphRunner":
        from .graph import ReactionGraphRunner

        return ReactionGraphRunner
    raise AttributeError(name)
