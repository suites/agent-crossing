import datetime
import json
import re
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


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\w+", text.lower()) if token}


def _semantic_similarity_proxy(a: str, b: str) -> float:
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = len(tokens_a.intersection(tokens_b))
    union = len(tokens_a.union(tokens_b))
    if union == 0:
        return 0.0
    return intersection / union


def _semantic_repeat_rate(
    *,
    session_history: list[tuple[str, str]],
    window: int = 4,
    threshold: float = 0.8,
) -> float:
    if len(session_history) < 2:
        return 0.0

    repeats = 0
    for index, (_, reply) in enumerate(session_history):
        if not reply.strip() or index == 0:
            continue
        recent = [text for _, text in session_history[max(0, index - window) : index]]
        if any(_semantic_similarity_proxy(reply, prev) >= threshold for prev in recent):
            repeats += 1

    total = max(1, len(session_history))
    return repeats / total


def _topic_progress_rate(session_history: list[tuple[str, str]]) -> float:
    if len(session_history) < 2:
        return 0.0

    progressed = 0
    evaluated = 0
    previous_tokens: set[str] = set()

    for _, reply in session_history:
        if not reply.strip():
            continue
        current_tokens = _tokenize(reply)
        if not current_tokens:
            continue
        evaluated += 1
        if not previous_tokens:
            progressed += 1
        else:
            new_ratio = len(current_tokens - previous_tokens) / max(1, len(current_tokens))
            if new_ratio >= 0.35 or "?" in reply:
                progressed += 1
        previous_tokens = current_tokens

    if evaluated == 0:
        return 0.0
    return progressed / evaluated


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
    speaker_name: str,
    partner_name: str,
    incoming_partner_utterance: str | None,
) -> list[str]:
    if incoming_partner_utterance and incoming_partner_utterance.strip():
        if language == "ko":
            return [f"{partner_name}의 직전 발화를 들음: {incoming_partner_utterance}"]
        return [
            f"Heard {partner_name}'s latest utterance: {incoming_partner_utterance}"
        ]

    if language == "ko":
        return [f"{speaker_name}가 {partner_name}를 근처에서 마주쳤다."]
    return [f"{speaker_name} encountered {partner_name} nearby."]


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

    initiator = agents[0]
    partner = agents[1]

    for turn in range(1, config.turns + 1):
        speaker = session.next_speaker()
        speaking_partner = partner if speaker is initiator else initiator

        incoming_partner_utterance = session.consume_incoming_partner_utterance(
            speaker=speaker
        )

        now = current_time + datetime.timedelta(seconds=config.turn_time_step_seconds)
        observed_events = _build_turn_observed_events(
            language=language,
            speaker_name=speaker.name,
            partner_name=speaking_partner.name,
            incoming_partner_utterance=incoming_partner_utterance,
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
        recent_replies_for_echo_check = _recent_replies_for_echo_check(
            session_history=session.history,
            window=config.repetition_window,
        )

        suppress_reason = ""
        if config.suppress_repeated_replies and _is_repetitive_reply(
            reply,
            recent_replies_for_echo_check,
        ):
            suppress_reason = "repeat_echo_suppressed"
            reply = ""

        fallback_reason = ""
        if not reply and config.fallback_on_empty_reply:
            fallback_reason = "empty_reply_fallback"
            reply = _fallback_reply(language)

        trace = dict(action_result.reaction_trace or {})
        if not trace.get("parse_success", True):
            parse_failures += 1
        if suppress_reason:
            trace["suppress_reason"] = suppress_reason
        if fallback_reason:
            trace["fallback_reason"] = fallback_reason

        print(f"[{turn:02d}] [THOUGHT] {speaker.name}: {action_result.thought}")
        print(f"[{turn:02d}] [ACTION] {speaker.name}: {action_result.action_summary}")
        print(
            f"[{turn:02d}] [DECISION_TRACE] {speaker.name}: "
            + json.dumps(trace, ensure_ascii=False)
        )

        if not reply:
            silent_turns += 1
            silent_reason = ",".join(
                reason
                for reason in [action_result.silent_reason, suppress_reason]
                if reason
            )
            if not silent_reason:
                silent_reason = "unknown"
            print(f"[{turn:02d}] [SILENT] {speaker.name} reason={silent_reason}")
            current_time = now
            continue

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

    parse_failure_rate = parse_failures / max(1, config.turns)
    silent_rate = silent_turns / max(1, config.turns)
    semantic_repeat_rate = _semantic_repeat_rate(session_history=session.history)
    topic_progress_rate = _topic_progress_rate(session.history)
    print("\nSimulation metrics")
    print(f"- parse_failure_rate={parse_failure_rate:.3f}")
    print(f"- silent_rate={silent_rate:.3f}")
    print(f"- semantic_repeat_rate={semantic_repeat_rate:.3f}")
    print(f"- topic_progress_rate={topic_progress_rate:.3f}")


def main() -> None:
    run_simulation(config=DEFAULT_CONFIG)


def _recent_replies_for_echo_check(
    *,
    session_history: list[tuple[str, str]],
    window: int,
) -> list[str]:
    if window < 1:
        return []
    return [reply for _, reply in session_history[-window:]]


if __name__ == "__main__":
    main()
