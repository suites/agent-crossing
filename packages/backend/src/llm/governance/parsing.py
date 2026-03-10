import json
import datetime
from dataclasses import dataclass, replace
from typing import Callable, TypeVar, cast

from agents.planning.models import DayPlanItem, HourlyPlanItem, MinutePlanItem
from llm.clients.ollama import JsonObject

from .contracts import (
    ReactionDecision,
    ReactionDecisionTrace,
    ReactionIntent,
    ReactionUtterance,
)


@dataclass(frozen=True)
class DayPlanParseResult:
    items: list[DayPlanItem]


class DayPlanParseError(ValueError):
    reason: str

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class HourPlanParseResult:
    items: list[HourlyPlanItem]


class HourPlanParseError(ValueError):
    reason: str

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class MinutePlanParseResult:
    items: list[MinutePlanItem]


class MinutePlanParseError(ValueError):
    reason: str

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def try_parse_day_plan(
    response_text: str,
    *,
    min_items: int = 5,
    max_items: int = 8,
) -> DayPlanParseResult:
    payload = parse_json_object(response_text)
    if payload is None:
        payload = attempt_json_repair_once(response_text)
    if payload is None:
        raise DayPlanParseError("json_parse_error_or_non_object")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise DayPlanParseError("missing_or_invalid_items")

    normalized = _normalize_day_plan_items(cast(list[object], raw_items))
    if len(normalized) > max_items:
        normalized = normalized[:max_items]
    if len(normalized) < min_items:
        raise DayPlanParseError("insufficient_day_plan_items")

    normalized.sort(key=lambda item: item.start_time)
    return DayPlanParseResult(items=normalized)


def try_parse_hour_plan(
    response_text: str,
    *,
    min_items: int = 1,
    max_items: int = 24,
) -> HourPlanParseResult:
    payload = parse_json_object(response_text)
    if payload is None:
        payload = attempt_json_repair_once(response_text)
    if payload is None:
        raise HourPlanParseError("json_parse_error_or_non_object")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise HourPlanParseError("missing_or_invalid_items")

    normalized = _normalize_plan_items(
        raw_items=cast(list[object], raw_items),
        item_factory=HourlyPlanItem,
        min_duration=1,
        duration_step=60,
        require_exact_hour=True,
    )
    if len(normalized) > max_items:
        normalized = normalized[:max_items]
    if len(normalized) < min_items:
        raise HourPlanParseError("insufficient_hour_plan_items")

    normalized.sort(key=lambda item: item.start_time)
    return HourPlanParseResult(items=normalized)


def try_parse_minute_plan(
    response_text: str,
    *,
    min_items: int = 1,
    max_items: int = 500,
) -> MinutePlanParseResult:
    payload = parse_json_object(response_text)
    if payload is None:
        payload = attempt_json_repair_once(response_text)
    if payload is None:
        raise MinutePlanParseError("json_parse_error_or_non_object")

    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise MinutePlanParseError("missing_or_invalid_items")

    normalized = _normalize_plan_items(
        raw_items=cast(list[object], raw_items),
        item_factory=MinutePlanItem,
        min_duration=5,
        max_duration=15,
        require_exact_minute=True,
    )
    if len(normalized) > max_items:
        normalized = normalized[:max_items]
    if len(normalized) < min_items:
        raise MinutePlanParseError("insufficient_minute_plan_items")

    normalized.sort(key=lambda item: item.start_time)
    return MinutePlanParseResult(items=normalized)


def _normalize_day_plan_items(raw_items: list[object]) -> list[DayPlanItem]:
    normalized: list[DayPlanItem] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue

        payload = cast(JsonObject, raw_item)

        start_time = _parse_iso_datetime(payload.get("start_time"))
        if start_time is None or not _is_exact_hour(start_time):
            continue

        duration_minutes = payload.get("duration_minutes")
        if (
            not isinstance(duration_minutes, int)
            or duration_minutes <= 0
            or duration_minutes % 60 != 0
        ):
            continue

        location = payload.get("location")
        if not isinstance(location, str) or not location.strip():
            continue

        action_content = payload.get("action_content")
        if not isinstance(action_content, str) or not action_content.strip():
            continue

        normalized_location = location.strip()
        normalized_action_content = action_content.strip()
        dedupe_key = (
            start_time.isoformat(),
            normalized_location.casefold(),
            normalized_action_content.casefold(),
        )
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized.append(
            DayPlanItem(
                start_time=start_time,
                duration_minutes=duration_minutes,
                location=normalized_location,
                action_content=normalized_action_content,
            )
        )

    return normalized


def _parse_iso_datetime(raw_value: object) -> datetime.datetime | None:
    if not isinstance(raw_value, str):
        return None

    value = raw_value.strip()
    if not value:
        return None

    candidate = value.replace("Z", "+00:00")
    try:
        return datetime.datetime.fromisoformat(candidate)
    except ValueError:
        return None


def parse_reaction_decision(response_text: str) -> ReactionDecision:
    default_trace = ReactionDecisionTrace(
        raw_response=response_text,
        parse_success=False,
        parse_error="json_parse_error_or_non_object",
        fallback_reason="parse_failure",
    )
    default_value = ReactionDecision(
        should_react=False,
        reaction="",
        reason="fallback",
        thought="",
        critique="",
        trace=default_trace,
    )
    parsed_json = parse_json_object(response_text)
    repaired_once = False
    if parsed_json is None:
        repaired_payload = attempt_json_repair_once(response_text)
        if repaired_payload is not None:
            parsed_json = repaired_payload
            repaired_once = True
        else:
            return default_value

    raw_should_react = parsed_json.get("should_react")
    if not isinstance(raw_should_react, bool):
        return replace(
            default_value,
            trace=replace(default_trace, parse_error="missing_or_invalid_should_react"),
        )

    raw_utterance = parsed_json.get("utterance")
    if not isinstance(raw_utterance, str):
        raw_utterance = ""

    raw_reaction = parsed_json.get("reaction")
    if not isinstance(raw_reaction, str):
        raw_reaction = ""
    final_reaction = raw_utterance.strip() or raw_reaction.strip()

    raw_thought = parsed_json.get("thought")
    if not isinstance(raw_thought, str):
        raw_thought = ""

    raw_critique = parsed_json.get("critique")
    if not isinstance(raw_critique, str):
        raw_critique = ""

    raw_reason = parsed_json.get("reason")
    if not isinstance(raw_reason, str):
        raw_reason = raw_critique or raw_thought or ""

    return ReactionDecision(
        should_react=raw_should_react,
        reaction=final_reaction,
        reason=raw_reason.strip() or "n/a",
        thought=raw_thought.strip(),
        critique=raw_critique.strip(),
        trace=ReactionDecisionTrace(
            raw_response=response_text,
            parse_success=True,
            parse_error="repaired_once" if repaired_once else "",
        ),
    )


def parse_reaction_intent(response_text: str) -> ReactionIntent:
    default_trace = ReactionDecisionTrace(
        raw_response=response_text,
        parse_success=False,
        parse_error="json_parse_error_or_non_object",
        fallback_reason="parse_failure",
    )
    default_value = ReactionIntent(
        should_react=False,
        reason="fallback",
        thought="",
        critique="",
        trace=default_trace,
    )
    parsed_json = parse_json_object(response_text)
    repaired_once = False
    if parsed_json is None:
        repaired_payload = attempt_json_repair_once(response_text)
        if repaired_payload is not None:
            parsed_json = repaired_payload
            repaired_once = True
        else:
            return default_value

    raw_should_react = parsed_json.get("should_react")
    if not isinstance(raw_should_react, bool):
        return replace(
            default_value,
            trace=replace(default_trace, parse_error="missing_or_invalid_should_react"),
        )

    raw_thought = parsed_json.get("thought")
    if not isinstance(raw_thought, str):
        raw_thought = ""

    raw_critique = parsed_json.get("critique")
    if not isinstance(raw_critique, str):
        raw_critique = ""

    raw_reason = parsed_json.get("reason")
    if not isinstance(raw_reason, str):
        raw_reason = raw_critique or raw_thought or ""

    return ReactionIntent(
        should_react=raw_should_react,
        reason=raw_reason.strip() or "n/a",
        thought=raw_thought.strip(),
        critique=raw_critique.strip(),
        trace=ReactionDecisionTrace(
            raw_response=response_text,
            parse_success=True,
            parse_error="repaired_once" if repaired_once else "",
        ),
    )


def parse_reaction_utterance(response_text: str) -> ReactionUtterance:
    default_trace = ReactionDecisionTrace(
        raw_response=response_text,
        parse_success=False,
        parse_error="json_parse_error_or_non_object",
        fallback_reason="parse_failure",
    )
    default_value = ReactionUtterance(
        utterance="",
        reason="fallback",
        thought="",
        critique="",
        trace=default_trace,
    )
    parsed_json = parse_json_object(response_text)
    repaired_once = False
    if parsed_json is None:
        repaired_payload = attempt_json_repair_once(response_text)
        if repaired_payload is not None:
            parsed_json = repaired_payload
            repaired_once = True
        else:
            return default_value

    raw_utterance = parsed_json.get("utterance")
    if not isinstance(raw_utterance, str):
        raw_utterance = ""

    raw_reaction = parsed_json.get("reaction")
    if not isinstance(raw_reaction, str):
        raw_reaction = ""
    final_utterance = raw_utterance.strip() or raw_reaction.strip()

    raw_thought = parsed_json.get("thought")
    if not isinstance(raw_thought, str):
        raw_thought = ""

    raw_critique = parsed_json.get("critique")
    if not isinstance(raw_critique, str):
        raw_critique = ""

    raw_reason = parsed_json.get("reason")
    if not isinstance(raw_reason, str):
        raw_reason = raw_critique or raw_thought or ""

    return ReactionUtterance(
        utterance=final_utterance,
        reason=raw_reason.strip() or "n/a",
        thought=raw_thought.strip(),
        critique=raw_critique.strip(),
        trace=ReactionDecisionTrace(
            raw_response=response_text,
            parse_success=True,
            parse_error="repaired_once" if repaired_once else "",
        ),
    )


TPlan = TypeVar("TPlan", DayPlanItem, HourlyPlanItem, MinutePlanItem)


def _normalize_plan_items(
    *,
    raw_items: list[object],
    item_factory: Callable[[datetime.datetime, int, str, str], TPlan],
    min_duration: int,
    max_duration: int | None = None,
    duration_step: int | None = None,
    require_exact_hour: bool = False,
    require_exact_minute: bool = False,
) -> list[TPlan]:
    normalized: list[TPlan] = []
    seen: set[tuple[str, str, str]] = set()

    for raw_item in raw_items:
        if not isinstance(raw_item, dict):
            continue

        payload = cast(JsonObject, raw_item)

        start_time = _parse_iso_datetime(payload.get("start_time"))
        if start_time is None:
            continue
        if require_exact_hour and not _is_exact_hour(start_time):
            continue
        if require_exact_minute and not _is_exact_minute(start_time):
            continue

        duration_minutes = _parse_duration_int(
            payload.get("duration_minutes"),
            min_duration=min_duration,
            max_duration=max_duration,
            step=duration_step,
        )
        if duration_minutes is None:
            continue

        location = payload.get("location")
        if not isinstance(location, str) or not location.strip():
            continue

        action_content = payload.get("action_content")
        if not isinstance(action_content, str) or not action_content.strip():
            continue

        normalized_location = location.strip()
        normalized_action_content = action_content.strip()
        dedupe_key = (
            start_time.isoformat(),
            normalized_location.casefold(),
            normalized_action_content.casefold(),
        )
        if dedupe_key in seen:
            continue

        seen.add(dedupe_key)
        normalized.append(
            item_factory(
                start_time,
                duration_minutes,
                normalized_location,
                normalized_action_content,
            )
        )

    return normalized


def _parse_duration_int(
    raw_value: object,
    *,
    min_duration: int,
    max_duration: int | None,
    step: int | None = None,
) -> int | None:
    if not isinstance(raw_value, int) or raw_value < min_duration:
        return None
    if max_duration is not None and raw_value > max_duration:
        return None
    if step is not None and raw_value % step != 0:
        return None
    return raw_value


def _is_exact_hour(value: datetime.datetime) -> bool:
    return value.minute == 0 and value.second == 0 and value.microsecond == 0


def _is_exact_minute(value: datetime.datetime) -> bool:
    return value.second == 0 and value.microsecond == 0


def parse_json_object(text: str) -> JsonObject | None:
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return cast(JsonObject, parsed)


def attempt_json_repair_once(text: str) -> JsonObject | None:
    candidate = text.strip()
    if not candidate:
        return None

    first_open = candidate.find("{")
    if first_open > 0:
        candidate = candidate[first_open:]

    last_close = candidate.rfind("}")
    if last_close >= 0:
        candidate = candidate[: last_close + 1]

    open_count = candidate.count("{")
    close_count = candidate.count("}")
    if open_count > close_count:
        candidate = candidate + ("}" * (open_count - close_count))

    return parse_json_object(candidate)
