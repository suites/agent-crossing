from dataclasses import dataclass
from typing import Literal


def fallback_reply(language: Literal["ko", "en"]) -> str:
    if language == "ko":
        return "LLM 응답 오류"
    return "LLM Error"


def normalize_reply_for_repeat_check(reply: str) -> str:
    normalized = " ".join(reply.lower().split())
    return normalized.strip(" .,!?:;\"'`()[]{}")


def is_repetitive_reply(reply: str, recent_replies: list[str]) -> bool:
    if not reply:
        return False

    normalized_reply = normalize_reply_for_repeat_check(reply)
    if not normalized_reply:
        return False

    normalized_recent_replies = {
        normalize_reply_for_repeat_check(recent_reply)
        for recent_reply in recent_replies
        if recent_reply.strip()
    }
    return normalized_reply in normalized_recent_replies


def recent_replies_for_echo_check(
    *,
    session_history: list[tuple[str, str]],
    window: int,
) -> list[str]:
    if window < 1:
        return []
    return [reply for _, reply in session_history[-window:]]


@dataclass(frozen=True)
class ReplyPolicyResult:
    reply: str
    suppress_reason: str
    fallback_reason: str


def apply_reply_policy(
    *,
    raw_reply: str,
    recent_replies: list[str],
    language: Literal["ko", "en"],
    suppress_repeated_replies: bool,
    fallback_on_empty_reply: bool,
) -> ReplyPolicyResult:
    reply = raw_reply.strip()
    suppress_reason = ""

    if suppress_repeated_replies and is_repetitive_reply(reply, recent_replies):
        suppress_reason = "repeat_echo_suppressed"
        reply = ""

    fallback_reason = ""
    if not reply and fallback_on_empty_reply:
        fallback_reason = "empty_reply_fallback"
        reply = fallback_reply(language)

    return ReplyPolicyResult(
        reply=reply,
        suppress_reason=suppress_reason,
        fallback_reason=fallback_reason,
    )
