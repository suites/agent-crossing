import datetime
from collections.abc import Sequence
from typing import Literal

from agents.agent import AgentIdentity, AgentProfile
from agents.memory.memory_object import MemoryObject


def build_retrieval_query(
    *,
    agent_identity: AgentIdentity,
    observation_content: str,
    dialogue_history: list[tuple[str, str]],
    profile: AgentProfile,
) -> str:
    lines: list[str] = _build_agent_context_lines(agent_identity, profile)
    lines.append(f"observation={observation_content}")

    if dialogue_history:
        partner_talk, my_talk = dialogue_history[-1]
        lines.append(f"recent_dialogue_partner={partner_talk}")
        lines.append(f"recent_dialogue_self={my_talk}")

    lines.append("task=상황판단에 필요한 기억 검색")
    return "\n".join(lines)


def build_reaction_prompt(
    *,
    agent_identity: AgentIdentity,
    current_time: datetime.datetime,
    observation_content: str,
    dialogue_history: list[tuple[str, str]],
    profile: AgentProfile,
    retrieved_memories: list[MemoryObject],
) -> str:
    summary_description = _build_summary_description(agent_identity, profile)
    agent_status = _build_agent_status(profile)
    memory_summary = _summarize_retrieved_memories(retrieved_memories)

    sections: list[str] = [
        "[Agent's Summary Description]",
        summary_description,
        f"It is {current_time.isoformat()}.",
        f"[{agent_identity.name}]'s status: {agent_status}.",
        f"Observation: {observation_content}",
    ]

    if dialogue_history:
        partner_talk, my_talk = dialogue_history[-1]
        sections.append("Recent dialogue context:")
        sections.append(f"- partner: {partner_talk or 'none'}")
        sections.append(f"- self: {my_talk or 'none'}")

    sections.extend(
        [
            f"Summary of relevant context from [{agent_identity.name}]'s memory:",
            memory_summary,
            (
                f"Should [{agent_identity.name}] react to the observation, "
                "and if so, what would be an appropriate reaction?"
            ),
        ]
    )

    return "\n".join(sections)


def build_reaction_decision_prompt(
    *,
    agent_identity: AgentIdentity,
    current_time: datetime.datetime,
    observation_content: str,
    dialogue_history: list[tuple[str, str]],
    profile: AgentProfile,
    retrieved_memories: list[MemoryObject],
) -> str:
    summary_description = _build_summary_description(agent_identity, profile)
    agent_status = _build_agent_status(profile)
    reflection_anchor = _build_reflection_anchor(profile, retrieved_memories)
    memory_summary = _summarize_retrieved_memories(retrieved_memories)

    sections: list[str] = [
        "[Agent's Summary Description]",
        summary_description,
        "[Identity Anchor - highest priority]",
        reflection_anchor,
        f"It is {current_time.isoformat()}.",
        f"[{agent_identity.name}]'s status: {agent_status}.",
        f"Observation: {observation_content}",
    ]

    if dialogue_history:
        sections.append("Recent dialogue context:")
        for index, (partner_talk, my_talk) in enumerate(dialogue_history, start=1):
            sections.append(f"- turn {index} partner: {partner_talk or 'none'}")
            sections.append(f"- turn {index} self: {my_talk or 'none'}")

    sections.extend(
        [
            (f"Summary of relevant context from [{agent_identity.name}]'s memory:"),
            memory_summary,
            (
                "If you provide a reaction, it must be spoken dialogue addressed "
                "to the conversation partner, not inner monologue."
            ),
            (
                "Keep utterance concise and short: ideally one sentence, no more "
                "than 80 Korean characters or 20 English words."
            ),
            (
                "Do not narrate personal schedules or plans unless saying them "
                "directly to the partner in natural conversation."
            ),
            (
                "When there is no prior dialogue context, use a brief greeting "
                "only when social context requires it. Avoid repetitive greeting "
                "phrases across turns."
            ),
            "Few-shot calibration examples:",
            _few_shot_reaction_examples(),
            (
                f"Should [{agent_identity.name}] react to the observation, "
                "and if so, what would be an appropriate reaction?"
            ),
            (
                "Return JSON only with this shape: "
                + '{"should_react": <boolean>, "thought": "<string>", '
                + '"critique": "<string>", "utterance": "<string>", '
                + '"reason": "<short string>"}'
            ),
        ]
    )

    return "\n".join(sections)


def language_system_prompt(language: Literal["ko", "en"]) -> str:
    if language == "ko":
        return (
            "You are simulating a conversational human agent. "
            "All generated natural-language reaction text must be in Korean only."
        )

    return (
        "You are simulating a conversational human agent. "
        "All generated natural-language reaction text must be in English only."
    )


def build_overlap_guard_block(
    *,
    recent_sentences: Sequence[str],
    previous_candidate: str,
) -> str:
    lines = [
        "Your previous reaction was too similar to recent dialogue.",
        "Generate a different reaction while preserving intent.",
        "Constraint: n-gram overlap with each sentence below must be <= 50%.",
        f"Previous candidate: {previous_candidate}",
        "Recent dialogue sentences:",
    ]
    for index, sentence in enumerate(recent_sentences, start=1):
        lines.append(f"- {index}. {sentence}")
    lines.append(
        "Return JSON only with this shape: "
        + '{"should_react": <boolean>, "thought": "<string>", '
        + '"critique": "<string>", "utterance": "<string>", '
        + '"reason": "<short string>"}'
    )
    return "\n".join(lines)


def build_semantic_guard_block(
    *,
    semantic_history: Sequence[str],
    previous_candidate: str,
    max_similarity: float,
    trigger: str,
    soft_threshold: float,
    hard_threshold: float,
) -> str:
    level = "hard block" if trigger == "hard" else "soft penalty"
    lines = [
        f"Your previous reaction violated semantic repetition guard ({level}).",
        f"max_similarity={max_similarity:.3f}, soft={soft_threshold}, hard={hard_threshold}",
        "Generate a meaningfully different utterance while keeping conversation natural.",
        f"Previous candidate: {previous_candidate}",
        "Recent self utterances to avoid semantically repeating:",
    ]
    for index, sentence in enumerate(semantic_history, start=1):
        lines.append(f"- {index}. {sentence}")
    lines.append(
        "Return JSON only with this shape: "
        + '{"should_react": <boolean>, "thought": "<string>", '
        + '"critique": "<string>", "utterance": "<string>", '
        + '"reason": "<short string>"}'
    )
    return "\n".join(lines)


def build_partner_response_nudge_block(*, latest_partner_utterance: str) -> str:
    return "\n".join(
        [
            "The partner has just spoken directly to you.",
            "Prefer a brief, natural reply instead of silence unless there is a strong social reason to stay silent.",
            f"Latest partner utterance: {latest_partner_utterance}",
            "If you still choose silence, reason must explicitly explain why.",
            "Return JSON only with this shape: "
            + '{"should_react": <boolean>, "thought": "<string>", '
            + '"critique": "<string>", "utterance": "<string>", '
            + '"reason": "<short string>"}',
        ]
    )


def _build_agent_context_lines(
    agent_identity: AgentIdentity,
    profile: AgentProfile,
) -> list[str]:
    lines: list[str] = [
        f"agent={agent_identity.name}",
        f"traits={', '.join(agent_identity.traits)}",
    ]

    if profile.fixed.identity_stable_set:
        lines.append(
            "identity_stable_set=" + " | ".join(profile.fixed.identity_stable_set[:2])
        )

    if profile.extended.current_plan_context:
        lines.append(
            "current_plan_context="
            + " | ".join(profile.extended.current_plan_context[:2])
        )

    return lines


def _build_summary_description(
    agent_identity: AgentIdentity,
    profile: AgentProfile,
) -> str:
    summary_lines: list[str] = [
        f"Name: {agent_identity.name}",
        f"Age: {agent_identity.age}",
        f"Traits: {', '.join(agent_identity.traits)}",
    ]

    if profile.fixed.identity_stable_set:
        summary_lines.append(
            "Identity stable set: " + " | ".join(profile.fixed.identity_stable_set[:3])
        )

    if profile.extended.lifestyle_and_routine:
        summary_lines.append(
            "Lifestyle and routine: "
            + " | ".join(profile.extended.lifestyle_and_routine[:2])
        )

    if profile.extended.current_plan_context:
        summary_lines.append(
            "Current plan context: "
            + " | ".join(profile.extended.current_plan_context[:2])
        )

    return "\n".join(summary_lines)


def _build_agent_status(profile: AgentProfile) -> str:
    if profile.extended.current_plan_context:
        return profile.extended.current_plan_context[0]
    return "Idle"


def _summarize_retrieved_memories(retrieved_memories: list[MemoryObject]) -> str:
    if not retrieved_memories:
        return "- no relevant memory found"

    lines: list[str] = []
    for index, memory in enumerate(retrieved_memories[:5], start=1):
        lines.append(f"- ({index}) [importance={memory.importance}] {memory.content}")
    return "\n".join(lines)


def _build_reflection_anchor(
    profile: AgentProfile,
    retrieved_memories: list[MemoryObject],
) -> str:
    reflection_items = [
        memory.content
        for memory in retrieved_memories
        if memory.node_type.value == "REFLECTION" and memory.content.strip()
    ]
    if reflection_items:
        return " | ".join(reflection_items[:2])

    if profile.fixed.identity_stable_set:
        return " | ".join(profile.fixed.identity_stable_set[:2])

    return "Keep consistency with your core identity and current plan."


def _few_shot_reaction_examples() -> str:
    return "\n".join(
        [
            "Example 1 (conflict with identity/plan -> polite refusal):",
            "- input: partner asks you to betray your stated values for convenience",
            '{"should_react": true, "utterance": "그건 제 원칙과 맞지 않아서 도와드리기 어려워요.", "thought": "정체성과 충돌", "critique": "정중히 거절", "reason": "identity_conflict"}',
            "Example 2 (natural pivot to own interest):",
            "- input: partner asks a vague small-talk question during your focused routine",
            '{"should_react": true, "utterance": "짧게는 괜찮아요. 저는 요즘 디카프 추출 실험이 더 궁금해요.", "thought": "관심사로 전환", "critique": "과잉 협조 대신 자연스러운 화제 전환", "reason": "natural_topic_shift"}',
        ]
    )
