import datetime
from dataclasses import dataclass
from typing import Literal

import numpy as np

from agents.agent import AgentProfile
from agents.memory.memory_object import MemoryObject
from agents.reaction import ReactionDecisionTrace

from ..decision_diagnostics import ActionDiagnostics


@dataclass(frozen=True)
class Observation:
    content: str
    now: datetime.datetime
    embedding: np.ndarray
    agent_name: str
    current_plan: str | None
    importance: int | None


@dataclass(frozen=True)
class DetermineContext:
    observation: Observation
    retrieved_memories: list[MemoryObject]
    dialogue_history: list[tuple[str, str]]
    profile: AgentProfile
    language: Literal["ko", "en"]


@dataclass(frozen=True)
class ActionLoopInput:
    current_time: datetime.datetime
    """시스템의 현재 시간."""
    dialogue_history: list[tuple[str, str]]
    """대화 상황에서, (상대방 발화, 나의 발화) 리스트. 가장 최근 발화가 리스트의 마지막에 위치한다."""
    profile: AgentProfile
    language: Literal["ko", "en"] = "ko"
    world_context: dict[str, str] | None = None
    observed_entities: list[str] | None = None
    observed_events: list[str] | None = None


@dataclass(frozen=True)
class ActionLoopResult:
    current_time: datetime.datetime
    """시스템의 현재 시간."""
    talk: str | None
    """Agent이 대화할 상황에서 생성된 대화 내용. 대화가 필요하지 않은 상황에서는 None."""
    utterance: str | None = None
    """대화 발화 결과. talk 필드와 동일한 값을 유지한다."""
    speak_decision: bool = False
    """이번 턴에 실제 발화를 수행했는지 여부."""
    action_intent: str = "continue_current_plan"
    """행동 의도를 나타내는 구조화된 문자열."""
    silent_reason: str = ""
    """발화하지 않았을 때 원인."""
    reaction_trace: ReactionDecisionTrace | None = None
    """LLM governance에서 생성한 reaction 추적 정보."""
    diagnostics: ActionDiagnostics | None = None
    """행동 판단 관측용 진단 정보."""
