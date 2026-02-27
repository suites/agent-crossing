from agents.agent import AgentIdentity, AgentProfile


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
        lines.append("recent_dialogue_context=")
        for index, (partner_talk, my_talk) in enumerate(dialogue_history, start=1):
            lines.append(f"turn_{index}_partner={partner_talk or 'none'}")
            lines.append(f"turn_{index}_self={my_talk or 'none'}")

    lines.append("task=상황판단에 필요한 기억 검색")
    return "\n".join(lines)


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
