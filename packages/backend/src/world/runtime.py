import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.sim_agent import SimAgent
from llm.governance import (
    ConversationMetrics,
    build_conversation_metrics,
)
from llm.clients.provider_factory import ProviderName, build_provider_client
from agents.world_factory import init_agents
from llm.clients.ollama import OllamaGenerateOptions

from .engine import SimulationEngine, SimulationEngineConfig, SimulationStepResult
from .session import WorldConversationSession


@dataclass(frozen=True)
class WorldRuntimeConfig:
    agent_persona_names: list[str]
    llm_provider: ProviderName
    base_url: str | None
    api_key: str | None
    llm_model: str
    embedding_model: str
    timeout_seconds: float
    persona_dir: str
    dialogue_turn_window: int | None = None
    language: Literal["ko", "en"] = "ko"
    fallback_on_empty_reply: bool = False
    suppress_repeated_replies: bool = True
    repetition_window: int = 4
    turn_time_step_seconds: int = 45
    reaction_generation_options: OllamaGenerateOptions = field(
        default_factory=lambda: OllamaGenerateOptions(
            temperature=0.35,
            top_p=0.92,
            num_predict=192,
            repeat_penalty=1.1,
            presence_penalty=0.2,
            frequency_penalty=0.4,
        )
    )


@dataclass(frozen=True)
class WorldRuntimeState:
    turn: int
    current_time: datetime.datetime
    parse_failures: int
    silent_turns: int
    history_size: int


class WorldRuntime:
    def __init__(
        self,
        *,
        agents: list[SimAgent],
        session: WorldConversationSession,
        engine: SimulationEngine,
        current_time: datetime.datetime,
    ) -> None:
        if len(agents) != 2:
            raise ValueError("WorldRuntime currently supports exactly two agents")

        self.agents: list[SimAgent] = agents
        self.session: WorldConversationSession = session
        self.engine: SimulationEngine = engine
        self.current_time: datetime.datetime = current_time
        self.turn: int = 0
        self.parse_failures: int = 0
        self.silent_turns: int = 0
        self._initiator: SimAgent = agents[0]
        self._partner: SimAgent = agents[1]

    def step(self) -> SimulationStepResult:
        self.turn += 1
        speaker = self.session.next_speaker()
        speaking_partner = (
            self._partner if speaker is self._initiator else self._initiator
        )
        step_result = self.engine.step(
            turn=self.turn,
            current_time=self.current_time,
            speaker=speaker,
            speaking_partner=speaking_partner,
        )
        self.current_time = step_result.now
        if step_result.parse_failure:
            self.parse_failures += 1
        if not step_result.reply:
            self.silent_turns += 1
        return step_result

    def metrics(self) -> ConversationMetrics:
        return build_conversation_metrics(
            turns=self.turn,
            parse_failures=self.parse_failures,
            silent_turns=self.silent_turns,
            session_history=self.session.history,
        )

    def state(self) -> WorldRuntimeState:
        return WorldRuntimeState(
            turn=self.turn,
            current_time=self.current_time,
            parse_failures=self.parse_failures,
            silent_turns=self.silent_turns,
            history_size=len(self.session.history),
        )


def build_world_runtime(*, config: WorldRuntimeConfig) -> WorldRuntime:
    now = datetime.datetime.now()
    llm_client = build_provider_client(
        provider=config.llm_provider,
        timeout_seconds=config.timeout_seconds,
        generation_model=config.llm_model,
        embedding_model=config.embedding_model,
        base_url=config.base_url,
        api_key=config.api_key,
    )
    agents = init_agents(
        persona_dir=config.persona_dir,
        agent_persona_names=config.agent_persona_names,
        llm_client=llm_client,
        embedding_model=config.embedding_model,
        now=now,
    )
    session = WorldConversationSession(
        agents=agents,
        dialogue_turn_window=config.dialogue_turn_window,
    )
    engine = SimulationEngine(
        session=session,
        config=SimulationEngineConfig(
            language=config.language,
            turn_time_step_seconds=config.turn_time_step_seconds,
            suppress_repeated_replies=config.suppress_repeated_replies,
            repetition_window=config.repetition_window,
            fallback_on_empty_reply=config.fallback_on_empty_reply,
            reaction_generation_options=config.reaction_generation_options,
        ),
    )
    return WorldRuntime(
        agents=agents,
        session=session,
        engine=engine,
        current_time=now,
    )


def default_persona_dir() -> str:
    return str(Path(__file__).resolve().parents[2] / "persona")
