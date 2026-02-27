import datetime
from dataclasses import dataclass
from importlib import import_module
from typing import Literal, Protocol, cast

import numpy as np
from agents.agent import AgentIdentity, AgentProfile
from llm.embedding_encoder import EmbeddingEncodingContext
from llm.llm_service import LlmService, ReactionDecision, ReactionDecisionInput

from .memory.memory_object import MemoryObject
from .memory.memory_service import MemoryService, ObservationContext
from .reflection_service import ReflectionService


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


@dataclass(frozen=True)
class ActionLoopResult:
    current_time: datetime.datetime
    """시스템의 현재 시간."""
    talk: str | None
    """Agent이 대화할 상황에서 생성된 대화 내용. 대화가 필요하지 않은 상황에서는 None."""


class PromptBuildersModule(Protocol):
    def build_retrieval_query(
        self,
        *,
        agent_identity: AgentIdentity,
        observation_content: str,
        dialogue_history: list[tuple[str, str]],
        profile: AgentProfile,
    ) -> str: ...


class AgentBrain:
    """Agent의 인지, 상황판단, 행동결정 등을 담당하는 핵심 클래스."""

    def __init__(
        self,
        *,
        agent_identity: AgentIdentity,
        memory_service: MemoryService,
        reflection_service: ReflectionService,
        llm_service: LlmService,
    ):
        self.memory_service: MemoryService = memory_service
        self.reflection_service: ReflectionService = reflection_service
        self.llm_service: LlmService = llm_service
        self.agent_identity: AgentIdentity = agent_identity

    def queue_observation(
        self,
        *,
        content: str,
        now: datetime.datetime,
        profile: AgentProfile,
        current_plan: str | None = None,
        importance: int | None = None,
    ) -> None:
        if current_plan is not None:
            final_current_plan = current_plan
        elif profile.extended.current_plan_context:
            final_current_plan = profile.extended.current_plan_context[0]
        else:
            final_current_plan = None
        memory = self.memory_service.create_observation_from_text(
            content=content,
            now=now,
            context=ObservationContext(
                agent_name=self.agent_identity.name,
                identity_stable_set=profile.fixed.identity_stable_set,
                current_plan=final_current_plan,
            ),
            importance=importance,
        )
        self.reflection_service.record_observation_importance(
            importance=memory.importance
        )

    def ingest_seed_memory(
        self,
        *,
        content: str,
        now: datetime.datetime,
        importance: int,
        identity_stable_set: list[str],
        current_plan: str | None,
    ) -> None:
        memory = self.memory_service.create_observation_from_text(
            content=content,
            now=now,
            context=ObservationContext(
                agent_name=self.agent_identity.name,
                identity_stable_set=identity_stable_set,
                current_plan=current_plan,
            ),
            importance=importance,
        )
        self.reflection_service.record_observation_importance(
            importance=memory.importance
        )

    def action_loop(self, input: ActionLoopInput) -> ActionLoopResult:
        """
        AgentBrain의 메인 루프.
        """
        # 1. 현재 상황을 인지한다. 인지할때 월드에서 현재 상황을 조회해서 주입한다.
        observation = self.perceive(
            now=input.current_time,
            current_plan_context=input.profile.extended.current_plan_context,
        )

        # 2. 인지된 정보들을 observation으로 메모리에 저장 (reflection 조건 충족 시 reflection도 함께 저장)
        self._save_observation_memory(observation, input.profile)
        if self.reflection_service.should_reflect():
            self.reflection_service.reflect(now=input.current_time)

        # 3. 상황판단을 한다.
        determine_result = self._determine_reaction(
            current_time=input.current_time,
            observation=observation,
            dialogue_history=input.dialogue_history,
            profile=input.profile,
            language=input.language,
        )

        # 4. 상황판단에 따라 반응을 결정한다.
        reaction_decision = self._react(determine_result)

        # 5. 반응에 때라 구체적인 행동 및 출력을 한다.
        return self._action(
            current_time=input.current_time,
            profile=input.profile,
            reaction_decision=reaction_decision,
        )

    def perceive(
        self,
        *,
        now: datetime.datetime,
        current_plan_context: list[str],
        world_context: dict[str, str] | None = None,
        observed_entities: list[str] | None = None,
        observed_events: list[str] | None = None,
    ) -> Observation:
        """
        현재 상황을 인지한다.
        """

        # 1) 시야 범위 내 핵심 상태를 관찰 단위로 정리한다.
        current_plan = current_plan_context[0] if current_plan_context else None
        context = world_context or {}

        lines: list[str] = []
        lines.append(f"time={now.isoformat()}")
        lines.append(f"agent={self.agent_identity.name}")
        lines.append(f"current_plan={current_plan or 'none'}")
        lines.append(f"traits={', '.join(self.agent_identity.traits)}")
        lines.append(f"location={context.get('location', 'unknown')}")

        entity_text = ", ".join(observed_entities) if observed_entities else "none"
        lines.append(f"entities={entity_text}")

        if observed_events:
            lines.append("events=" + "; ".join(observed_events))
        else:
            lines.append("events=none")

        content = "\n".join(lines)

        embedding = self.memory_service.embedding_encoder.encode(
            EmbeddingEncodingContext(text=content)
        )
        return Observation(
            content=content,
            now=now,
            embedding=embedding,
            agent_name=self.agent_identity.name,
            current_plan=current_plan,
            importance=None,
        )

    def _action(
        self,
        *,
        current_time: datetime.datetime,
        profile: AgentProfile,
        reaction_decision: ReactionDecision,
    ) -> ActionLoopResult:
        # 5. 구체적인 행동 결정 및 출력을 한다.
        talk = reaction_decision.reaction or None

        if talk is not None:
            self.queue_observation(
                content=f"I decided to react: {talk}",
                now=current_time,
                profile=profile,
            )

        return ActionLoopResult(
            current_time=current_time,
            talk=talk,
        )

    def _determine_reaction(
        self,
        *,
        current_time: datetime.datetime,
        observation: Observation,
        dialogue_history: list[tuple[str, str]],
        profile: AgentProfile,
        language: Literal["ko", "en"],
    ) -> DetermineContext:
        # 1. 상황판단을 위해 기억을 검색한다.
        retrieval_query = self._build_retrieval_query(
            observation=observation,
            dialogue_history=dialogue_history,
            profile=profile,
        )
        retrieved_memories = self.memory_service.get_retrieval_memories(
            query=retrieval_query, current_time=current_time
        )
        return DetermineContext(
            observation=observation,
            dialogue_history=dialogue_history,
            profile=profile,
            retrieved_memories=retrieved_memories,
            language=language,
        )

    def _build_retrieval_query(
        self,
        *,
        observation: Observation,
        dialogue_history: list[tuple[str, str]],
        profile: AgentProfile,
    ) -> str:
        prompt_builders_module = cast(
            PromptBuildersModule,
            cast(object, import_module("agents.prompt_builders")),
        )
        return prompt_builders_module.build_retrieval_query(
            agent_identity=self.agent_identity,
            observation_content=observation.content,
            dialogue_history=dialogue_history,
            profile=profile,
        )

    def _react(self, determine_context: DetermineContext) -> ReactionDecision:
        # 4. 상황판단에 따라 반응을 결정한다.
        # 4-1. 관찰 결과가 단순하다면 기존 계획을 수행한다.
        # 4-2. 관찰 결과가 중요하거나 예상치 못하면 기존 계획을 멈추고 반응한다.
        # 4-2-1. 반응하기로 결정했으면 행동 계획을 재생성하고, 대화중이라면 자연어 대화도 생성한다.
        return self.llm_service.decide_reaction(
            ReactionDecisionInput(
                agent_identity=self.agent_identity,
                current_time=determine_context.observation.now,
                observation_content=determine_context.observation.content,
                dialogue_history=determine_context.dialogue_history,
                profile=determine_context.profile,
                retrieved_memories=determine_context.retrieved_memories,
                language=determine_context.language,
            )
        )

    def _save_observation_memory(
        self,
        observation: Observation,
        profile: AgentProfile,
    ) -> None:
        """
        메모:
        - 목적: observation 저장 + (조건 충족 시) reflection 엔트리 호출.
        - 입력/출력: 관찰 입력값들 -> (observation 1개, reflection 리스트)
        """
        memory = self.memory_service.create_observation(
            content=observation.content,
            now=observation.now,
            embedding=observation.embedding,
            context=ObservationContext(
                agent_name=observation.agent_name,
                identity_stable_set=profile.fixed.identity_stable_set,
                current_plan=observation.current_plan,
            ),
            importance=observation.importance,
        )

        self.reflection_service.record_observation_importance(
            importance=memory.importance
        )
