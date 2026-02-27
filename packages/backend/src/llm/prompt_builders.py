import datetime

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
