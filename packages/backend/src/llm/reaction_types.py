import datetime
from dataclasses import dataclass
from typing import Literal, Protocol

from agents.agent import AgentIdentity, AgentProfile
from agents.memory.memory_object import MemoryObject
from llm.ollama_client import OllamaGenerateOptions


@dataclass(frozen=True)
class ReactionDecisionInput:
    agent_identity: AgentIdentity
    current_time: datetime.datetime
    observation_content: str
    dialogue_history: list[tuple[str, str]]
    profile: AgentProfile
    retrieved_memories: list[MemoryObject]
    language: Literal["ko", "en"] = "ko"


@dataclass(frozen=True)
class ReactionDecisionTrace:
    raw_response: str
    parse_success: bool
    parse_error: str = ""
    fallback_reason: str = ""
    suppress_reason: str = ""
    overlap_retry_count: int = 0
    partner_retry_count: int = 0
    semantic_retry_count: int = 0
    max_semantic_similarity: float = 0.0
    semantic_hard_threshold: float = 0.92
    semantic_soft_threshold: float = 0.82
    semantic_retry_trigger: str = "none"


@dataclass(frozen=True)
class ReactionDecision:
    should_react: bool
    reaction: str
    reason: str
    thought: str = ""
    critique: str = ""
    trace: ReactionDecisionTrace = ReactionDecisionTrace(
        raw_response="",
        parse_success=False,
        parse_error="uninitialized",
    )


class GenerateClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        options: OllamaGenerateOptions | None = None,
        format_json: bool = False,
    ) -> str: ...
