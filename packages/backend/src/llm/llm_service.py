import datetime
import json
import re
from dataclasses import dataclass
from typing import Literal, Protocol, cast

from agents.agent import AgentIdentity, AgentProfile
from agents.memory.memory_object import MemoryObject
from llm.ollama_client import JsonObject, OllamaGenerateOptions


class LlmService:
    def __init__(self, ollama_client: "GenerateClient"):
        self.ollama_client: GenerateClient = ollama_client

    def generate_salient_high_level_questions(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list[str]:
        if not memories:
            return []

        # 1. 기억 나열
        memory_lines = [f"Statements about {agent_name}"]
        for i, memory in enumerate(memories):
            memory_lines.append(f"{i + 1}. {memory.content}")

        memory_text = "\n".join(memory_lines)

        # 2. 핵심 지시어 + JSON 포맷 강제
        instruction = (
            "Given only the information above, what are 3 most salient high-level "
            "questions we can answer about the subjects in the statements?\n"
            "Return JSON only with this shape: "
            '{"questions": ["<question 1>", "<question 2>", "<question 3>"]}'
        )

        prompt = f"{memory_text}\n\n{instruction}"

        # 3. LLM 호출 및 파싱
        response_text = self.ollama_client.generate(prompt=prompt)

        try:
            parsed_data = cast(object, json.loads(response_text))
            if not isinstance(parsed_data, dict):
                return []

            parsed_payload = cast(JsonObject, parsed_data)
            questions = parsed_payload.get("questions", [])
            if not isinstance(questions, list):
                return []

            question_items = cast(list[object], questions)
            parsed_questions: list[str] = []
            for question in question_items:
                if isinstance(question, str) and question.strip():
                    parsed_questions.append(question)

            return parsed_questions
        except json.JSONDecodeError:
            # 포맷이 깨졌을 경우의 방어 코드
            return []

    def generate_insights_with_citation_key(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list["InsightWithCitation"]:
        if not memories:
            return []

        # 1. 기억 나열
        memory_lines = [f"Statements about {agent_name}"]
        for i, memory in enumerate(memories):
            memory_lines.append(f"{i + 1}. {memory.content}")

        memory_text = "\n".join(memory_lines)
        statement_to_memory_id = {
            index + 1: memory.id for index, memory in enumerate(memories)
        }

        instruction = (
            "What 5 high-level insights can you infer from the above statements? "
            "Use statement numbers as evidence references.\n"
            "Return JSON only with this shape: "
            '{"insights": ['
            '{"insight": "<text>", "citation_statement_numbers": [1, 5, 3]}, '
            '{"insight": "<text>", "citation_statement_numbers": [2, 4]}'
            "]}"
        )

        prompt = f"{memory_text}\n\n{instruction}"

        # 3. LLM 호출 및 파싱
        response_text = self.ollama_client.generate(prompt=prompt, format_json=True)

        try:
            parsed_data = cast(object, json.loads(response_text))
            if not isinstance(parsed_data, dict):
                return []

            parsed_payload = cast(JsonObject, parsed_data)
            insights = parsed_payload.get("insights", [])
            if not isinstance(insights, list):
                return []

            insight_items = cast(list[object], insights)
            parsed_insights: list[InsightWithCitation] = []
            for raw_insight in insight_items:
                if not isinstance(raw_insight, dict):
                    continue

                insight_payload = cast(JsonObject, raw_insight)
                insight_text = insight_payload.get("insight")
                citation_numbers = insight_payload.get("citation_statement_numbers")
                if not isinstance(insight_text, str) or not insight_text.strip():
                    continue

                citation_memory_ids: list[int] = []
                if isinstance(citation_numbers, list):
                    citation_number_items = cast(list[object], citation_numbers)
                    for raw_number in citation_number_items:
                        if not isinstance(raw_number, int):
                            continue

                        citation_memory_id = statement_to_memory_id.get(raw_number)
                        if citation_memory_id is not None:
                            citation_memory_ids.append(citation_memory_id)

                parsed_insights.append(
                    InsightWithCitation(
                        context=insight_text.strip(),
                        citation_memory_ids=citation_memory_ids,
                    )
                )

            return parsed_insights
        except json.JSONDecodeError:
            # 포맷이 깨졌을 경우의 방어 코드
            return []

    def decide_reaction(
        self,
        input: "ReactionDecisionInput",
        *,
        generation_options: OllamaGenerateOptions | None = None,
    ) -> "ReactionDecision":
        prompt = self._build_reaction_decision_prompt(input)
        system_prompt = _language_system_prompt(input.language)
        recent_sentences = _recent_dialogue_sentences(
            input.dialogue_history,
            window=3,
        )

        retry_count = 0
        max_retry = 2
        working_prompt = prompt

        while True:
            response_text = self.ollama_client.generate(
                prompt=working_prompt,
                system=system_prompt,
                options=generation_options,
                format_json=True,
            )
            decision = _parse_reaction_decision(response_text)

            if (
                not decision.should_react
                or not decision.reaction
                or not recent_sentences
                or retry_count >= max_retry
            ):
                return decision

            has_overlap = _exceeds_ngram_overlap_threshold(
                candidate_sentence=decision.reaction,
                recent_sentences=recent_sentences,
                n=2,
                threshold=0.5,
            )
            if not has_overlap:
                return decision

            retry_count += 1
            overlap_guard = _build_overlap_guard_block(
                recent_sentences=recent_sentences,
                previous_candidate=decision.reaction,
            )
            working_prompt = f"{prompt}\n\n{overlap_guard}"

    @staticmethod
    def _build_reaction_decision_prompt(input: "ReactionDecisionInput") -> str:
        summary_description = _build_summary_description(
            input.agent_identity,
            input.profile,
        )
        agent_status = _build_agent_status(input.profile)
        memory_summary = _summarize_retrieved_memories(input.retrieved_memories)

        sections: list[str] = [
            "[Agent's Summary Description]",
            summary_description,
            f"It is {input.current_time.isoformat()}.",
            f"[{input.agent_identity.name}]'s status: {agent_status}.",
            f"Observation: {input.observation_content}",
        ]

        if input.dialogue_history:
            sections.append("Recent dialogue context:")
            for index, (partner_talk, my_talk) in enumerate(
                input.dialogue_history, start=1
            ):
                sections.append(f"- turn {index} partner: {partner_talk or 'none'}")
                sections.append(f"- turn {index} self: {my_talk or 'none'}")

        sections.extend(
            [
                (
                    "Summary of relevant context from "
                    f"[{input.agent_identity.name}]'s memory:"
                ),
                memory_summary,
                (
                    "If you provide a reaction, it must be spoken dialogue addressed "
                    "to the conversation partner, not inner monologue."
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
                (
                    f"Should [{input.agent_identity.name}] react to the observation, "
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


def _parse_reaction_decision(response_text: str) -> "ReactionDecision":
    default_value = ReactionDecision(
        should_react=False,
        reaction="",
        reason="fallback",
        thought="",
        critique="",
    )
    parsed_json = _parse_json_object(response_text)
    if parsed_json is None:
        return default_value

    raw_should_react = parsed_json.get("should_react")
    if not isinstance(raw_should_react, bool):
        return default_value

    raw_utterance = parsed_json.get("utterance")
    if not isinstance(raw_utterance, str):
        raw_utterance = ""

    raw_reaction = parsed_json.get("reaction")
    if not isinstance(raw_reaction, str):
        raw_reaction = ""
    final_reaction = raw_utterance.strip() or raw_reaction.strip()

    raw_thought = parsed_json.get("thought")
    if not isinstance(raw_thought, str):
        raw_thought = ""

    raw_critique = parsed_json.get("critique")
    if not isinstance(raw_critique, str):
        raw_critique = ""

    raw_reason = parsed_json.get("reason")
    if not isinstance(raw_reason, str):
        raw_reason = raw_critique or raw_thought or ""

    return ReactionDecision(
        should_react=raw_should_react,
        reaction=final_reaction,
        reason=raw_reason.strip() or "n/a",
        thought=raw_thought.strip(),
        critique=raw_critique.strip(),
    )


def _parse_json_object(text: str) -> JsonObject | None:
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return cast(JsonObject, parsed)


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


def _language_system_prompt(language: Literal["ko", "en"]) -> str:
    if language == "ko":
        return (
            "You are simulating a conversational human agent. "
            "All generated natural-language reaction text must be in Korean only."
        )

    return (
        "You are simulating a conversational human agent. "
        "All generated natural-language reaction text must be in English only."
    )


def _recent_dialogue_sentences(
    dialogue_history: list[tuple[str, str]],
    *,
    window: int,
) -> list[str]:
    if window < 1:
        return []

    ordered_sentences: list[str] = []
    for partner_talk, my_talk in dialogue_history:
        if partner_talk and partner_talk.strip() and partner_talk.strip() != "none":
            ordered_sentences.append(partner_talk.strip())
        if my_talk and my_talk.strip() and my_talk.strip() != "none":
            ordered_sentences.append(my_talk.strip())

    return ordered_sentences[-window:]


def _tokenize_for_ngram(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _sentence_ngrams(sentence: str, n: int) -> set[tuple[str, ...]]:
    tokens = _tokenize_for_ngram(sentence)
    if not tokens:
        return set()

    if len(tokens) < n:
        n = 1

    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _overlap_ratio(
    candidate_sentence: str, reference_sentence: str, *, n: int
) -> float:
    candidate_ngrams = _sentence_ngrams(candidate_sentence, n)
    if not candidate_ngrams:
        return 0.0

    reference_ngrams = _sentence_ngrams(reference_sentence, n)
    if not reference_ngrams:
        return 0.0

    overlap_count = len(candidate_ngrams.intersection(reference_ngrams))
    return overlap_count / len(candidate_ngrams)


def _exceeds_ngram_overlap_threshold(
    *,
    candidate_sentence: str,
    recent_sentences: list[str],
    n: int,
    threshold: float,
) -> bool:
    for recent_sentence in recent_sentences:
        if _overlap_ratio(candidate_sentence, recent_sentence, n=n) > threshold:
            return True
    return False


def _build_overlap_guard_block(
    *,
    recent_sentences: list[str],
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


@dataclass(frozen=True)
class InsightWithCitation:
    context: str
    citation_memory_ids: list[int]


@dataclass(frozen=True)
class ReactionDecisionInput:
    agent_identity: AgentIdentity
    current_time: datetime.datetime
    observation_content: str
    dialogue_history: list[tuple[str, str]]
    profile: AgentProfile
    retrieved_memories: list[MemoryObject]
    language: Literal["ko", "en"] = "ko"


@dataclass(frozen=True)
class ReactionDecision:
    should_react: bool
    reaction: str
    reason: str
    thought: str = ""
    critique: str = ""


class GenerateClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        options: OllamaGenerateOptions | None = None,
        format_json: bool = False,
    ) -> str: ...
