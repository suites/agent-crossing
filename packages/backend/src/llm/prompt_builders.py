import datetime
from collections.abc import Sequence
from typing import Literal

from agents.agent import AgentIdentity, AgentProfile
from agents.memory.memory_object import MemoryObject
from agents.planning.models import DayPlanItem, HourlyPlanItem

from .template_loader import render_template

REACTION_INTENT_JSON_SHAPE = (
    '{"should_react": <boolean>, "thought": "<string>", '
    '"critique": "<string>", "reason": "<short string>"}'
)

REACTION_UTTERANCE_JSON_SHAPE = (
    '{"utterance": "<string>", "thought": "<string>", '
    '"critique": "<string>", "reason": "<short string>"}'
)

SALIENT_QUESTIONS_JSON_SHAPE = (
    '{"questions": ["<question 1>", "<question 2>", "<question 3>"]}'
)

INSIGHTS_JSON_SHAPE = (
    '{"insights": ['
    '{"insight": "<text>", "citation_statement_numbers": [1, 5, 3]}, '
    '{"insight": "<text>", "citation_statement_numbers": [2, 4]}'
    "]}"
)

IMPORTANCE_JSON_SHAPE = '{"importance": <int 1-10>, "reason": "<short>"}'

DAY_PLAN_JSON_SHAPE = (
    '{"items": ['
    '{"start_time": "<ISO-8601 datetime>", "duration_minutes": <positive int>, '
    '"location": "<location>", "action_content": "<action text>"}'
    "]}"
)

HOURLY_PLAN_JSON_SHAPE = (
    '{"items": ['
    '{"start_time": "<ISO-8601 datetime>", "duration_minutes": <positive int>, '
    '"location": "<location>", "action_content": "<action text>"}'
    "]}"
)

MINUTE_PLAN_JSON_SHAPE = (
    '{"items": ['
    '{"start_time": "<ISO-8601 datetime>", "duration_minutes": <int 5-15>, '
    '"location": "<location>", "action_content": "<action text>"}'
    "]}"
)


def build_salient_questions_prompt(
    *,
    agent_name: str,
    memories: list[MemoryObject],
) -> str:
    memory_text = _build_memory_statements_text(
        agent_name=agent_name, memories=memories
    )
    instruction = render_template(
        "salient_questions_instruction.md",
        json_shape=SALIENT_QUESTIONS_JSON_SHAPE,
    )
    return f"{memory_text}\n\n{instruction.strip()}"


def build_insights_with_citation_prompt(
    *,
    agent_name: str,
    memories: list[MemoryObject],
) -> str:
    memory_text = _build_memory_statements_text(
        agent_name=agent_name, memories=memories
    )
    instruction = render_template(
        "insights_instruction.md",
        json_shape=INSIGHTS_JSON_SHAPE,
    )
    return f"{memory_text}\n\n{instruction.strip()}"


def build_importance_scoring_prompt(
    *,
    agent_name: str,
    identity_stable_set: list[str],
    current_plan: str | None,
    observation: str,
) -> str:
    identity_text = " | ".join(identity_stable_set[:3]) or "N/A"
    current_plan_text = current_plan or "N/A"
    return render_template(
        "importance_scoring.md",
        json_shape=IMPORTANCE_JSON_SHAPE,
        agent_name=agent_name,
        identity_text=identity_text,
        current_plan_text=current_plan_text,
        observation=observation,
    )


def build_day_plan_prompt(
    *,
    agent_name: str,
    age: int,
    innate_traits: list[str],
    persona_background: str,
    yesterday_date_text: str,
    yesterday_summary: str,
    today_date_text: str,
) -> str:
    """Build a persona-grounded prompt for daily structured plan generation."""
    traits_text = ", ".join(trait.strip() for trait in innate_traits if trait.strip())
    return render_template(
        "day_plan_broad_strokes_instruction.md",
        agent_name=agent_name,
        age=str(age),
        innate_traits=traits_text or "N/A",
        persona_background=persona_background.strip(),
        yesterday_date_text=yesterday_date_text.strip(),
        yesterday_summary=yesterday_summary.strip(),
        today_date_text=today_date_text.strip(),
        json_shape=DAY_PLAN_JSON_SHAPE,
    )


def build_hourly_plan_prompt(
    *,
    agent_name: str,
    today_date_text: str,
    day_plan_items: list[DayPlanItem],
) -> str:
    day_plan_lines = "\n".join(
        f"- {item.start_time.isoformat()} ({item.duration_minutes}m) | {item.location} | {item.action_content}"
        for item in day_plan_items
    )
    if not day_plan_lines:
        day_plan_lines = "- no day plan items"

    return render_template(
        "hourly_plan_instruction.md",
        agent_name=agent_name,
        today_date_text=today_date_text.strip(),
        day_plan_lines=day_plan_lines,
        json_shape=HOURLY_PLAN_JSON_SHAPE,
    )


def build_minute_plan_prompt(
    *,
    agent_name: str,
    current_time: datetime.datetime,
    hourly_plan_items: list[HourlyPlanItem],
) -> str:
    hourly_plan_lines = "\n".join(
        f"- {item.start_time.isoformat()} ({item.duration_minutes}m) | {item.location} | {item.action_content}"
        for item in hourly_plan_items
    )
    if not hourly_plan_lines:
        hourly_plan_lines = "- no hourly plan items"

    return render_template(
        "minute_plan_instruction.md",
        agent_name=agent_name,
        current_time=current_time.isoformat(),
        hourly_plan_lines=hourly_plan_lines,
        json_shape=MINUTE_PLAN_JSON_SHAPE,
    )


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

    sections: list[str] = _build_reaction_base_sections(
        agent_identity=agent_identity,
        current_time=current_time,
        summary_description=summary_description,
        agent_status=agent_status,
        observation_content=observation_content,
    )

    if dialogue_history:
        partner_talk, my_talk = dialogue_history[-1]
        sections.append("Recent dialogue context:")
        sections.append(f"- partner: {partner_talk or 'none'}")
        sections.append(f"- self: {my_talk or 'none'}")

    sections.extend(
        [
            f"Summary of relevant context from [{agent_identity.name}]'s memory:",
            memory_summary,
            _reaction_intent_question(agent_identity.name),
        ]
    )

    return "\n".join(sections)


def build_reaction_intent_prompt(
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

    sections: list[str] = _build_reaction_base_sections(
        agent_identity=agent_identity,
        current_time=current_time,
        summary_description=summary_description,
        agent_status=agent_status,
        observation_content=observation_content,
        identity_anchor=reflection_anchor,
    )

    if dialogue_history:
        sections.append("Recent dialogue context:")
        for index, (partner_talk, my_talk) in enumerate(dialogue_history, start=1):
            sections.append(f"- turn {index} partner: {partner_talk or 'none'}")
            sections.append(f"- turn {index} self: {my_talk or 'none'}")

    sections.extend(
        [
            (f"Summary of relevant context from [{agent_identity.name}]'s memory:"),
            memory_summary,
            _reaction_intent_question(agent_identity.name),
            _reaction_intent_shape_line(),
        ]
    )

    return "\n".join(sections)


def build_reaction_utterance_prompt(
    *,
    agent_identity: AgentIdentity,
    current_time: datetime.datetime,
    observation_content: str,
    dialogue_history: list[tuple[str, str]],
    profile: AgentProfile,
    retrieved_memories: list[MemoryObject],
    intent_reason: str,
    intent_thought: str,
    intent_critique: str,
) -> str:
    summary_description = _build_summary_description(agent_identity, profile)
    agent_status = _build_agent_status(profile)
    reflection_anchor = _build_reflection_anchor(profile, retrieved_memories)
    memory_summary = _summarize_retrieved_memories(retrieved_memories)

    sections: list[str] = _build_reaction_base_sections(
        agent_identity=agent_identity,
        current_time=current_time,
        summary_description=summary_description,
        agent_status=agent_status,
        observation_content=observation_content,
        identity_anchor=reflection_anchor,
    )

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
                "Stage 1 decision: should_react=true | "
                f"reason={intent_reason or 'n/a'} | "
                f"thought={intent_thought or 'n/a'} | "
                f"critique={intent_critique or 'n/a'}"
            ),
            render_template("reaction_guidelines.md").strip(),
            "Few-shot calibration examples:",
            _few_shot_reaction_examples(),
            _reaction_utterance_question(agent_identity.name),
            _reaction_utterance_shape_line(),
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
    return build_reaction_intent_prompt(
        agent_identity=agent_identity,
        current_time=current_time,
        observation_content=observation_content,
        dialogue_history=dialogue_history,
        profile=profile,
        retrieved_memories=retrieved_memories,
    )


def language_system_prompt(language: Literal["ko", "en"]) -> str:
    if language == "ko":
        return render_template("language_system_ko.md").strip()

    return render_template("language_system_en.md").strip()


def build_overlap_guard_block(
    *,
    recent_sentences: Sequence[str],
    previous_candidate: str,
) -> str:
    recent_dialogue_lines = "\n".join(
        f"- {index}. {sentence}"
        for index, sentence in enumerate(recent_sentences, start=1)
    )
    return render_template(
        "overlap_guard.md",
        previous_candidate=previous_candidate,
        recent_dialogue_lines=recent_dialogue_lines,
        json_shape=REACTION_UTTERANCE_JSON_SHAPE,
    ).strip()


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
    semantic_history_lines = "\n".join(
        f"- {index}. {sentence}"
        for index, sentence in enumerate(semantic_history, start=1)
    )
    return render_template(
        "semantic_guard.md",
        level=level,
        max_similarity=f"{max_similarity:.3f}",
        soft_threshold=str(soft_threshold),
        hard_threshold=str(hard_threshold),
        previous_candidate=previous_candidate,
        semantic_history_lines=semantic_history_lines,
        json_shape=REACTION_UTTERANCE_JSON_SHAPE,
    ).strip()


def build_partner_response_nudge_block(*, latest_partner_utterance: str) -> str:
    return render_template(
        "partner_response_nudge.md",
        latest_partner_utterance=latest_partner_utterance,
        json_shape=REACTION_UTTERANCE_JSON_SHAPE,
    ).strip()


def _build_reaction_base_sections(
    *,
    agent_identity: AgentIdentity,
    current_time: datetime.datetime,
    summary_description: str,
    agent_status: str,
    observation_content: str,
    identity_anchor: str | None = None,
) -> list[str]:
    sections: list[str] = [
        "[Agent's Summary Description]",
        summary_description,
    ]
    if identity_anchor is not None:
        sections.extend(
            [
                "[Identity Anchor - highest priority]",
                identity_anchor,
            ]
        )

    sections.extend(
        [
            f"It is {current_time.isoformat()}.",
            f"[{agent_identity.name}]'s status: {agent_status}.",
            f"Observation: {observation_content}",
        ]
    )
    return sections


def _reaction_intent_question(agent_name: str) -> str:
    rendered = render_template(
        "reaction_intent_question.md",
        agent_name=agent_name,
        json_shape=REACTION_INTENT_JSON_SHAPE,
    ).strip()
    return _first_content_line(rendered)


def _reaction_intent_shape_line() -> str:
    rendered = render_template(
        "reaction_intent_question.md",
        agent_name="agent",
        json_shape=REACTION_INTENT_JSON_SHAPE,
    ).strip()
    for line in rendered.splitlines():
        if line.startswith("Return strict JSON only"):
            return line
    return (
        "Return strict JSON only with this exact shape and no extra text: "
        f"{REACTION_INTENT_JSON_SHAPE}"
    )


def _reaction_utterance_question(agent_name: str) -> str:
    rendered = render_template(
        "reaction_utterance_question.md",
        agent_name=agent_name,
        json_shape=REACTION_UTTERANCE_JSON_SHAPE,
    ).strip()
    return _first_content_line(rendered)


def _reaction_utterance_shape_line() -> str:
    rendered = render_template(
        "reaction_utterance_question.md",
        agent_name="agent",
        json_shape=REACTION_UTTERANCE_JSON_SHAPE,
    ).strip()
    for line in rendered.splitlines():
        if line.startswith("Return strict JSON only"):
            return line
    return (
        "Return strict JSON only with this exact shape and no extra text: "
        f"{REACTION_UTTERANCE_JSON_SHAPE}"
    )


def _first_content_line(rendered: str) -> str:
    for line in rendered.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("##"):
            continue
        return stripped
    return ""


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


def _build_memory_statements_text(
    *,
    agent_name: str,
    memories: list[MemoryObject],
) -> str:
    memory_lines = [f"Statements about {agent_name}"]
    for index, memory in enumerate(memories, start=1):
        memory_lines.append(f"{index}. {memory.content}")
    return "\n".join(memory_lines)


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
    return render_template("reaction_few_shot_examples.md").strip()


def template_file_plan() -> list[str]:
    return [
        "salient_questions_instruction.md",
        "insights_instruction.md",
        "importance_scoring.md",
        "day_plan_broad_strokes_instruction.md",
        "hourly_plan_instruction.md",
        "minute_plan_instruction.md",
        "reaction_guidelines.md",
        "reaction_few_shot_examples.md",
        "reaction_decision_question.md",
        "reaction_intent_question.md",
        "reaction_utterance_question.md",
        "language_system_ko.md",
        "language_system_en.md",
        "overlap_guard.md",
        "semantic_guard.md",
        "partner_response_nudge.md",
    ]
