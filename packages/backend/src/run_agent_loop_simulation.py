import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agents.agent_brain import ActionLoopInput
from agents.sim_agent import SimAgent
from agents.world_factory import init_agents
from llm.ollama_client import OllamaClient
from world.session import WorldConversationSession


def _fallback_reply(language: Literal["ko", "en"]) -> str:
    if language == "ko":
        return "LLM 응답 오류"
    return "LLM Error"


def _format_conversation_start_intent(
    language: Literal["ko", "en"],
    *,
    target_name: str,
) -> str:
    if language == "ko":
        return f"{target_name}에게 인사를 건네며 말을 걸기로 결정했다."
    return (
        f"I decided to initiate a conversation with {target_name}. "
        "I will begin with a greeting."
    )


def _format_self_said(language: Literal["ko", "en"], reply: str) -> str:
    if language == "ko":
        return f"나는 이렇게 말했다: {reply}"
    return f"I said: {reply}"


def _format_other_said(
    language: Literal["ko", "en"], speaker_name: str, reply: str
) -> str:
    if language == "ko":
        return f"{speaker_name}가 이렇게 말했다: {reply}"
    return f"{speaker_name} said: {reply}"


def _ingest_line(observer: SimAgent, content: str, now: datetime.datetime) -> None:
    observer.brain.queue_observation(
        content=content,
        now=now,
        profile=observer.profile,
    )


@dataclass(frozen=True)
class LoopSimulationConfig:
    agent_persona_names: list[str]
    turns: int
    base_url: str
    llm_model: str
    timeout_seconds: float
    persona_dir: str
    dialogue_turn_window: int = 10
    language: Literal["ko", "en"] = "ko"


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
    if len(agent_persona_names) < 2:
        raise ValueError("At least two agents are required")

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

    session = WorldConversationSession(
        agents=agents,
        dialogue_turn_window=config.dialogue_turn_window,
    )

    initiator = agents[0]
    partner = agents[1]
    _ingest_line(
        observer=initiator,
        content=_format_conversation_start_intent(language, target_name=partner.name),
        now=current_time,
    )

    for turn in range(1, config.turns + 1):
        speaker = session.next_speaker()

        incoming_partner_utterance = session.consume_incoming_partner_utterance(
            speaker=speaker
        )

        now = current_time
        action_result = speaker.brain.action_loop(
            ActionLoopInput(
                current_time=now,
                dialogue_history=session.dialogue_context_for(speaker=speaker),
                profile=speaker.profile,
                language=language,
            )
        )

        reply = (action_result.talk or "").strip()
        if not reply:
            reply = _fallback_reply(language)

        print(f"[{turn:02d}] [THOUGHT] {speaker.name}: {action_result.thought}")
        print(f"[{turn:02d}] [ACTION] {speaker.name}: {action_result.action_summary}")

        session.commit_speaker_reply(
            speaker=speaker,
            incoming_partner_utterance=incoming_partner_utterance,
            reply=reply,
        )
        print(f"{speaker.name}: {reply}")

        for observer in agents:
            if observer is speaker:
                _ingest_line(observer, _format_self_said(language, reply), now)
                continue

            _ingest_line(
                observer, _format_other_said(language, speaker.name, reply), now
            )
            session.incoming_utterances_by_agent[observer.name].append(reply)

        current_time = now

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
