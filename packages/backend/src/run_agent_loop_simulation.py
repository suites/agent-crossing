import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from llm.ollama_client import OllamaGenerateOptions
from world.runtime import WorldRuntimeConfig, build_world_runtime


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

    runtime = build_world_runtime(
        config=WorldRuntimeConfig(
            agent_persona_names=agent_persona_names,
            base_url=config.base_url,
            llm_model=config.llm_model,
            timeout_seconds=config.timeout_seconds,
            persona_dir=config.persona_dir,
            dialogue_turn_window=config.dialogue_turn_window,
            language=language,
            fallback_on_empty_reply=config.fallback_on_empty_reply,
            suppress_repeated_replies=config.suppress_repeated_replies,
            repetition_window=config.repetition_window,
            turn_time_step_seconds=config.turn_time_step_seconds,
            reaction_generation_options=config.reaction_generation_options,
        )
    )

    for turn in range(1, config.turns + 1):
        speaker = runtime.session.agents[(turn - 1) % len(runtime.session.agents)]
        step_result = runtime.step()

        print(f"[{turn:02d}] [THOUGHT] {speaker.name}: {step_result.thought}")
        print(f"[{turn:02d}] [ACTION] {speaker.name}: {step_result.action_summary}")
        print(
            f"[{turn:02d}] [DECISION_TRACE] {speaker.name}: "
            + json.dumps(step_result.trace, ensure_ascii=False)
        )

        if not step_result.reply:
            silent_reason = step_result.silent_reason or "unknown"
            print(f"[{turn:02d}] [SILENT] {speaker.name} reason={silent_reason}")
            continue

        print(f"{speaker.name}: {step_result.reply}")

    print("\nRecent memories")
    for agent in runtime.agents:
        print(f"\n- {agent.name}")
        memories = agent.memory_service.get_recent_memories(limit=5)
        for memory in memories:
            print(f"  [{memory.node_type.value}] {memory.content}")

    metrics = runtime.metrics()
    print("\nSimulation metrics")
    print(f"- parse_failure_rate={metrics.parse_failure_rate:.3f}")
    print(f"- silent_rate={metrics.silent_rate:.3f}")
    print(f"- semantic_repeat_rate={metrics.semantic_repeat_rate:.3f}")
    print(f"- topic_progress_rate={metrics.topic_progress_rate:.3f}")


def main() -> None:
    run_simulation(config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
