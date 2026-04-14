from __future__ import annotations

import datetime
import json
import os
import time
from collections.abc import Sequence
from dataclasses import asdict

from agents.planning.models import (
    DayPlanBroadStrokesRequest,
    DayPlanItem,
    HourlyPlanItem,
    MinutePlanItem,
)
from agents.planning.planner import Planner
from llm.clients.provider_factory import build_provider_client
from llm.llm_gateway import LlmGateway

GOOGLE_AI_STUDIO_API_KEY = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "").strip()
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "google_ai_studio")
GENERATION_MODEL = os.getenv("LLM_MODEL", "gemini-1.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")
TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))


PlanItem = DayPlanItem | HourlyPlanItem | MinutePlanItem


def _print_plan_items(title: str, items: Sequence[PlanItem]) -> None:
    payload: list[dict[str, str | int]] = []
    for item in items:
        serialized_item = asdict(item)
        serialized_item["start_time"] = item.start_time.isoformat(timespec="minutes")
        serialized_item["end_time"] = item.end_time.isoformat(timespec="minutes")
        payload.append(serialized_item)

    print(f"{title}:\n{json.dumps(payload, ensure_ascii=False, indent=2)}")


def test_planner_live_with_provider_factory() -> None:
    provider_client = build_provider_client(
        provider=LLM_PROVIDER,
        timeout_seconds=TIMEOUT_SECONDS,
        generation_model=GENERATION_MODEL,
        embedding_model=EMBEDDING_MODEL,
        api_key=GOOGLE_AI_STUDIO_API_KEY,
    )
    planner = Planner(LlmGateway(provider_client))

    day_plan_items = planner.generate_day_plan(
        DayPlanBroadStrokesRequest(
            agent_name="Eddy Lin",
            age=19,
            innate_traits=["friendly", "outgoing", "hospitable"],
            persona_background="Music theory student focusing on composition.",
            yesterday_date=datetime.datetime(2026, 2, 12),
            yesterday_summary="Studied harmony and practiced composition in the evening.",
            today_date=datetime.datetime(2026, 2, 13),
        )
    )

    _print_plan_items("Day Plan Items", day_plan_items)
    assert 5 <= len(day_plan_items) <= 8

    hourly_plan_time = day_plan_items[0].start_time + datetime.timedelta(minutes=30)

    time.sleep(1)

    hourly_plan_items = planner.generate_hourly_plan(
        agent_name="Eddy Lin",
        current_time=hourly_plan_time,
        day_plan_item=day_plan_items[0],
    )

    _print_plan_items("Hourly Plan Items", hourly_plan_items)
    assert len(hourly_plan_items) > 0

    minute_plan_time = hourly_plan_items[0].start_time + datetime.timedelta(minutes=10)

    time.sleep(1)

    minute_plan_items = planner.generate_minute_plan(
        agent_name="Eddy Lin",
        current_time=minute_plan_time,
        hourly_plan_item=hourly_plan_items[0],
    )

    _print_plan_items("Minute Plan Items", minute_plan_items)
    assert len(minute_plan_items) > 0
