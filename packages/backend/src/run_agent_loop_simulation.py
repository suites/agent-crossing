import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from agents.agent_brain import ActionLoopInput
from agents.sim_agent import SimAgent
from agents.world_factory import init_agents
from llm.ollama_client import OllamaClient, OllamaGenerateOptions
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


def _normalize_reply_for_repeat_check(reply: str) -> str:
    normalized = " ".join(reply.lower().split())
    return normalized.strip(" .,!?:;\"'`()[]{}")


def _is_repetitive_reply(reply: str, recent_replies: list[str]) -> bool:
    if not reply:
        return False

    normalized_reply = _normalize_reply_for_repeat_check(reply)
    if not normalized_reply:
        return False

    normalized_recent_replies = {
        _normalize_reply_for_repeat_check(recent_reply)
        for recent_reply in recent_replies
        if recent_reply.strip()
    }
    return normalized_reply in normalized_recent_replies


def _build_turn_world_context(
    *, speaker_name: str, partner_name: str, turn: int
) -> dict[str, str]:
    locations = [
        "town square",
        "cafe entrance",
        "library walkway",
        "park bench",
    ]
    location = locations[(turn - 1) % len(locations)]
    return {
        "location": f"{location} near {partner_name}",
        "focus": f"{speaker_name} is facing {partner_name}",
    }


def _build_turn_observed_events(
    *,
    language: Literal["ko", "en"],
    partner_name: str,
    incoming_partner_utterance: str | None,
    is_opening_turn: bool,
    turn: int,
) -> list[str]:
    if incoming_partner_utterance and incoming_partner_utterance.strip():
        if language == "ko":
            return [f"{partner_name}의 직전 발화를 들음: {incoming_partner_utterance}"]
        return [
            f"Heard {partner_name}'s latest utterance: {incoming_partner_utterance}"
        ]

    if is_opening_turn:
        if language == "ko":
            return [f"대화의 시작 구간이다 (turn={turn})."]
        return [f"Conversation opening phase (turn={turn})."]

    if language == "ko":
        return [f"{partner_name}를 근처에서 마주쳤고, 아직 직전 발화는 없다."]
    return [f"Met {partner_name} nearby, but there is no immediate prior utterance."]


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
    fallback_on_empty_reply: bool = False
    suppress_repeated_replies: bool = True
    repetition_window: int = 4
    turn_time_step_seconds: int = 45
    reaction_generation_options: OllamaGenerateOptions = field(
        default_factory=lambda: OllamaGenerateOptions(
            temperature=0.35,
            top_p=0.92,
            num_predict=80,
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

    session = WorldConversationSession(
        agents=agents,
        dialogue_turn_window=config.dialogue_turn_window,
    )

    initiator = agents[0]
    partner = agents[1]
    recent_replies_by_agent: dict[str, list[str]] = {agent.name: [] for agent in agents}
    _ingest_line(
        observer=initiator,
        content=_format_conversation_start_intent(language, target_name=partner.name),
        now=current_time,
    )

    for turn in range(1, config.turns + 1):
        speaker = session.next_speaker()
        speaking_partner = partner if speaker is initiator else initiator

        incoming_partner_utterance = session.consume_incoming_partner_utterance(
            speaker=speaker
        )

        now = current_time + datetime.timedelta(seconds=config.turn_time_step_seconds)
        observed_events = _build_turn_observed_events(
            language=language,
            partner_name=speaking_partner.name,
            incoming_partner_utterance=incoming_partner_utterance,
            is_opening_turn=(turn == 1 and speaker is initiator),
            turn=turn,
        )
        action_result = speaker.brain.action_loop(
            ActionLoopInput(
                current_time=now,
                dialogue_history=session.dialogue_context_for(speaker=speaker),
                profile=speaker.profile,
                language=language,
                world_context=_build_turn_world_context(
                    speaker_name=speaker.name,
                    partner_name=speaking_partner.name,
                    turn=turn,
                ),
                observed_entities=[speaking_partner.name],
                observed_events=observed_events,
                reaction_generation_options=config.reaction_generation_options,
            )
        )

        reply = (action_result.utterance or action_result.talk or "").strip()
        if config.suppress_repeated_replies and _is_repetitive_reply(
            reply,
            recent_replies_by_agent[speaker.name][-config.repetition_window :],
        ):
            reply = ""

        if not reply and config.fallback_on_empty_reply:
            reply = _fallback_reply(language)

        print(f"[{turn:02d}] [THOUGHT] {speaker.name}: {action_result.thought}")
        print(f"[{turn:02d}] [ACTION] {speaker.name}: {action_result.action_summary}")

        if not reply:
            print(f"[{turn:02d}] [SILENT] {speaker.name}")
            current_time = now
            continue

        session.commit_speaker_reply(
            speaker=speaker,
            incoming_partner_utterance=incoming_partner_utterance,
            reply=reply,
        )
        recent_replies_by_agent[speaker.name].append(reply)
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
