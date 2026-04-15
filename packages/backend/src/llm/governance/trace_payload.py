from dataclasses import asdict
from typing import cast

from agents.reaction.contracts import ReactionDecisionTrace


def reaction_trace_to_payload(
    *,
    trace: ReactionDecisionTrace | None,
) -> dict[str, object]:
    if trace is None:
        return {}
    return cast(dict[str, object], asdict(trace))


def merge_policy_trace(
    *,
    trace: ReactionDecisionTrace | None,
    suppress_reason: str,
    fallback_reason: str,
) -> dict[str, object]:
    payload = reaction_trace_to_payload(trace=trace)
    if suppress_reason:
        payload["suppress_reason"] = suppress_reason
    if fallback_reason:
        payload["fallback_reason"] = fallback_reason
    return payload


def is_reaction_parse_failure(*, trace: ReactionDecisionTrace | None) -> bool:
    if trace is None:
        return False
    return not trace.parse_success
