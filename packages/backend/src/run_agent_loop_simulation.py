import datetime
from dataclasses import dataclass
from pathlib import Path

from agents.agent_brain import ActionLoopInput
from agents.sim_agent import SimAgent
from agents.world_factory import init_agents
from llm.ollama_client import OllamaClient


def _fallback_reply() -> str:
    return "LLM Error"


@dataclass(frozen=True)
class LoopSimulationConfig:
    agent_persona_names: list[str]
    turns: int
    base_url: str
    llm_model: str
    timeout_seconds: float
    persona_dir: str


DEFAULT_CONFIG = LoopSimulationConfig(
    agent_persona_names=["Jiho", "Sujin"],
    turns=3,
    base_url="http://localhost:11434",
    llm_model="qwen2.5:7b-instruct",
    timeout_seconds=30.0,
    persona_dir=str(Path(__file__).resolve().parents[1] / "persona"),
)


def ingest_line(observer: SimAgent, content: str, now: datetime.datetime) -> None:
    observer.brain.queue_observation(
        content=content,
        now=now,
        profile=observer.profile,
    )


def run_simulation(
    *,
    config: LoopSimulationConfig,
) -> None:
    agent_persona_names = config.agent_persona_names
    if len(agent_persona_names) < 2:
        raise ValueError("At least two agents are required")

    ollama_client = OllamaClient(
        base_url=config.base_url,
        timeout_seconds=config.timeout_seconds,
    )
    agents = init_agents(
        persona_dir=config.persona_dir,
        agent_persona_names=agent_persona_names,
        ollama_client=ollama_client,
        llm_model=config.llm_model,
        now=datetime.datetime.now(),
    )

    history: list[tuple[str, str]] = []
    dialogue_history_by_agent: dict[str, list[tuple[str, str]]] = {
        agent.name: [] for agent in agents
    }
    incoming_utterances_by_agent: dict[str, list[str]] = {
        agent.name: [] for agent in agents
    }

    for turn in range(1, config.turns + 1):
        speaker = agents[(turn - 1) % len(agents)]

        incoming_partner_utterance: str | None = None
        incoming_queue = incoming_utterances_by_agent[speaker.name]
        if incoming_queue:
            incoming_partner_utterance = incoming_queue.pop(0)
            dialogue_history_by_agent[speaker.name].append(
                (incoming_partner_utterance, "")
            )

        now = datetime.datetime.now()
        action_result = speaker.brain.action_loop(
            ActionLoopInput(
                current_time=now,
                dialogue_history=dialogue_history_by_agent[speaker.name],
                profile=speaker.profile,
            )
        )

        reply = (action_result.talk or "").strip()
        if not reply:
            reply = _fallback_reply()

        if incoming_partner_utterance is not None:
            dialogue_history_by_agent[speaker.name][-1] = (
                incoming_partner_utterance,
                reply,
            )
        else:
            dialogue_history_by_agent[speaker.name].append(("", reply))

        history.append((speaker.name, reply))
        print(f"[{turn:02d}] {speaker.name}: {reply}")

        for observer in agents:
            if observer is speaker:
                ingest_line(observer, f"I said: {reply}", now)
                continue

            ingest_line(observer, f"{speaker.name} said: {reply}", now)
            incoming_utterances_by_agent[observer.name].append(reply)

    print("\nRecent memories")
    for agent in agents:
        print(f"\n- {agent.name}")
        memories = agent.memory_service.get_recent_memories(limit=5)
        for memory in memories:
            print(f"  [{memory.node_type.value}] {memory.content}")


def main() -> None:
    run_simulation(config=DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
