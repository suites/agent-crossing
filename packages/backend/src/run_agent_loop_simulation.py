import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.world_factory import init_agents
from llm.ollama_client import OllamaClient, OllamaGenerateOptions
from metrics.conversation_metrics import build_conversation_metrics
from world.engine import SimulationEngine, SimulationEngineConfig
from world.session import WorldConversationSession


@dataclass(frozen=True)
class LoopSimulationConfig:
    agent_persona_names: list[str]
    turns: int
    base_url: str
    llm_model: str
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


DEFAULT_CONFIG = LoopSimulationConfig(
    agent_persona_names=["Jiho", "Sujin"],
    turns=10,
    base_url="http://localhost:11434",
    llm_model="qwen2.5:7b-instruct",
    timeout_seconds=30.0,
    persona_dir=str(Path(__file__).resolve().parents[1] / "persona"),
)


def run_simulation(
    *,
    config: LoopSimulationConfig,
) -> None:
    language = config.language
    agent_persona_names = config.agent_persona_names
    if len(agent_persona_names) != 2:
        raise ValueError("This simulation currently supports exactly two agents")

    ollama_client = OllamaClient(
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
    )
    current_time = datetime.datetime.now()
    agents = init_agents(
        persona_dir=config.persona_dir,
        agent_persona_names=agent_persona_names,
        ollama_client=ollama_client,
        llm_model=config.llm_model,
        now=current_time,
    )

    parse_failures = 0
    silent_turns = 0

    session = WorldConversationSession(
        agents=agents,
        dialogue_turn_window=config.dialogue_turn_window,
    )
    engine = SimulationEngine(
        session=session,
        config=SimulationEngineConfig(
            language=language,
            turn_time_step_seconds=config.turn_time_step_seconds,
            suppress_repeated_replies=config.suppress_repeated_replies,
            repetition_window=config.repetition_window,
            fallback_on_empty_reply=config.fallback_on_empty_reply,
            reaction_generation_options=config.reaction_generation_options,
        ),
    )

    initiator = agents[0]
    partner = agents[1]

    for turn in range(1, config.turns + 1):
        speaker = session.next_speaker()
        speaking_partner = partner if speaker is initiator else initiator

        step_result = engine.step(
            turn=turn,
            current_time=current_time,
            speaker=speaker,
            speaking_partner=speaking_partner,
        )

        if step_result.parse_failure:
            parse_failures += 1

        print(f"[{turn:02d}] [THOUGHT] {speaker.name}: {step_result.thought}")
        print(f"[{turn:02d}] [ACTION] {speaker.name}: {step_result.action_summary}")
        print(
            f"[{turn:02d}] [DECISION_TRACE] {speaker.name}: "
            + json.dumps(step_result.trace, ensure_ascii=False)
        )

        if not step_result.reply:
            silent_turns += 1
            silent_reason = step_result.silent_reason or "unknown"
            print(f"[{turn:02d}] [SILENT] {speaker.name} reason={silent_reason}")
            current_time = step_result.now
            continue

        print(f"{speaker.name}: {step_result.reply}")
        current_time = step_result.now

    print("\nRecent memories")
    for agent in agents:
        print(f"\n- {agent.name}")
        memories = agent.memory_service.get_recent_memories(limit=5)
        for memory in memories:
            print(f"  [{memory.node_type.value}] {memory.content}")

    metrics = build_conversation_metrics(
        turns=config.turns,
        parse_failures=parse_failures,
        silent_turns=silent_turns,
        session_history=session.history,
    )
    print("\nSimulation metrics")
    print(f"- parse_failure_rate={metrics.parse_failure_rate:.3f}")
    print(f"- silent_rate={metrics.silent_rate:.3f}")
    print(f"- semantic_repeat_rate={metrics.semantic_repeat_rate:.3f}")
    print(f"- topic_progress_rate={metrics.topic_progress_rate:.3f}")


def main() -> None:
    run_simulation(config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
