import re
from dataclasses import dataclass
from typing import cast


def tokenize(text: str) -> set[str]:
    tokens = cast(list[str], re.findall(r"\w+", text.lower()))
    return {token for token in tokens if token}


def semantic_similarity_proxy(a: str, b: str) -> float:
    tokens_a = tokenize(a)
    tokens_b = tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return intersection / union


def semantic_repeat_rate(
    *,
    session_history: list[tuple[str, str]],
    window: int = 4,
    threshold: float = 0.8,
) -> float:
    if len(session_history) < 2:
        return 0.0

    repeats = 0
    for index, (_, reply) in enumerate(session_history):
        if not reply.strip() or index == 0:
            continue
        recent = [text for _, text in session_history[max(0, index - window) : index]]
        if any(semantic_similarity_proxy(reply, prev) >= threshold for prev in recent):
            repeats += 1

    total = max(1, len(session_history))
    return repeats / total


def topic_progress_rate(session_history: list[tuple[str, str]]) -> float:
    if len(session_history) < 2:
        return 0.0

    progressed = 0
    evaluated = 0
    previous_tokens: set[str] = set()

    for _, reply in session_history:
        if not reply.strip():
            continue
        current_tokens = tokenize(reply)
        if not current_tokens:
            continue
        evaluated += 1
        if not previous_tokens:
            progressed += 1
        else:
            new_ratio = len(current_tokens - previous_tokens) / max(
                1, len(current_tokens)
            )
            if new_ratio >= 0.35 or "?" in reply:
                progressed += 1
        previous_tokens = current_tokens

    if evaluated == 0:
        return 0.0
    return progressed / evaluated


@dataclass(frozen=True)
class ConversationMetrics:
    parse_failure_rate: float
    silent_rate: float
    semantic_repeat_rate: float
    topic_progress_rate: float


def build_conversation_metrics(
    *,
    turns: int,
    parse_failures: int,
    silent_turns: int,
    session_history: list[tuple[str, str]],
) -> ConversationMetrics:
    return ConversationMetrics(
        parse_failure_rate=parse_failures / max(1, turns),
        silent_rate=silent_turns / max(1, turns),
        semantic_repeat_rate=semantic_repeat_rate(session_history=session_history),
        topic_progress_rate=topic_progress_rate(session_history),
    )
