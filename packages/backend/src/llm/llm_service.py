import datetime
import json
import re
from dataclasses import dataclass, replace
from typing import Literal, Protocol, cast

from agents.agent import AgentIdentity, AgentProfile
from agents.memory.memory_object import MemoryObject
from llm.embedding_encoder import EmbeddingEncodingContext
from llm.ollama_client import JsonObject, OllamaGenerateOptions
from utils.math import cosine_similarity

SEMANTIC_HARD_BLOCK_THRESHOLD = 0.92
SEMANTIC_SOFT_PENALTY_THRESHOLD = 0.82


class LlmService:
    def __init__(
        self,
        ollama_client: "GenerateClient",
        *,
        embedding_encoder: "EmbeddingEncoder | None" = None,
    ):
        self.ollama_client: GenerateClient = ollama_client
        self.embedding_encoder = embedding_encoder

    def generate_salient_high_level_questions(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list[str]:
        if not memories:
            return []

        memory_lines = [f"Statements about {agent_name}"]
        for i, memory in enumerate(memories):
            memory_lines.append(f"{i + 1}. {memory.content}")

        memory_text = "\n".join(memory_lines)

        instruction = (
            "Given only the information above, what are 3 most salient high-level "
            "questions we can answer about the subjects in the statements?\n"
            "Return JSON only with this shape: "
            '{"questions": ["<question 1>", "<question 2>", "<question 3>"]}'
        )

        prompt = f"{memory_text}\n\n{instruction}"
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
            return []

    def generate_insights_with_citation_key(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list["InsightWithCitation"]:
        if not memories:
            return []

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
            return []

    def decide_reaction(
        self,
        input: "ReactionDecisionInput",
        *,
        generation_options: OllamaGenerateOptions | None = None,
    ) -> "ReactionDecision":
        prompt = self._build_reaction_decision_prompt(input)
        system_prompt = _language_system_prompt(input.language)
        recent_sentences = _recent_dialogue_sentences(input.dialogue_history, window=3)

        retry_count = 0
        max_retry = 2
        semantic_retry_count = 0
        max_semantic_retry = 2
        partner_retry_count = 0
        working_prompt = prompt
        latest_partner_utterance = _latest_partner_utterance(input.dialogue_history)

        semantic_history = _recent_self_utterances(input.dialogue_history, window=5)
        semantic_history_embeddings = _embed_sentences(
            sentences=semantic_history,
            embedding_encoder=self.embedding_encoder,
        )

        while True:
            response_text = self.ollama_client.generate(
                prompt=working_prompt,
                system=system_prompt,
                options=generation_options,
                format_json=True,
            )
            decision = _parse_reaction_decision(response_text)

            if (
                latest_partner_utterance
                and not decision.should_react
                and partner_retry_count < 1
            ):
                partner_retry_count += 1
                working_prompt = (
                    f"{prompt}\n\n"
                    + _build_partner_response_nudge_block(
                        latest_partner_utterance=latest_partner_utterance
                    )
                )
                continue

            decision = replace(
                decision,
                trace=replace(
                    decision.trace,
                    partner_retry_count=partner_retry_count,
                    semantic_hard_threshold=SEMANTIC_HARD_BLOCK_THRESHOLD,
                    semantic_soft_threshold=SEMANTIC_SOFT_PENALTY_THRESHOLD,
                ),
            )

            semantic_check = _semantic_overlap_check(
                candidate_sentence=decision.reaction,
                reference_sentences=semantic_history,
                reference_embeddings=semantic_history_embeddings,
                embedding_encoder=self.embedding_encoder,
            )

            if semantic_check.max_similarity >= SEMANTIC_SOFT_PENALTY_THRESHOLD:
                semantic_retry_count += 1
                if semantic_retry_count > max_semantic_retry:
                    return replace(
                        decision,
                        trace=replace(
                            decision.trace,
                            semantic_retry_count=semantic_retry_count,
                            max_semantic_similarity=semantic_check.max_similarity,
                            semantic_retry_trigger=semantic_check.trigger,
                            fallback_reason="semantic_retry_exhausted",
                        ),
                    )

                semantic_guard = _build_semantic_guard_block(
                    semantic_history=semantic_history,
                    previous_candidate=decision.reaction,
                    max_similarity=semantic_check.max_similarity,
                    trigger=semantic_check.trigger,
                )
                working_prompt = f"{prompt}\n\n{semantic_guard}"
                continue

            if not decision.should_react or not decision.reaction or not recent_sentences:
                return replace(
                    decision,
                    trace=replace(
                        decision.trace,
                        semantic_retry_count=semantic_retry_count,
                        max_semantic_similarity=semantic_check.max_similarity,
                        semantic_retry_trigger=semantic_check.trigger,
                    ),
                )

            has_overlap = _exceeds_ngram_overlap_threshold(
                candidate_sentence=decision.reaction,
                recent_sentences=recent_sentences,
                n=2,
                threshold=0.5,
            )
            if not has_overlap:
                return replace(
                    decision,
                    trace=replace(
                        decision.trace,
                        semantic_retry_count=semantic_retry_count,
                        max_semantic_similarity=semantic_check.max_similarity,
                        semantic_retry_trigger=semantic_check.trigger,
                    ),
                )

            retry_count += 1
            if retry_count > max_retry:
                return replace(
                    decision,
                    trace=replace(
                        decision.trace,
                        overlap_retry_count=retry_count,
                        semantic_retry_count=semantic_retry_count,
                        max_semantic_similarity=semantic_check.max_similarity,
                        semantic_retry_trigger=semantic_check.trigger,
                        fallback_reason="overlap_retry_exhausted",
                    ),
                )

            overlap_guard = _build_overlap_guard_block(
                recent_sentences=recent_sentences,
                previous_candidate=decision.reaction,
            )
            working_prompt = f"{prompt}\n\n{overlap_guard}"

    @staticmethod
    def _build_reaction_decision_prompt(input: "ReactionDecisionInput") -> str:
        summary_description = _build_summary_description(input.agent_identity, input.profile)
        agent_status = _build_agent_status(input.profile)
        memory_summary = _summarize_retrieved_memories(input.retrieved_memories)

        sections: list[str] = [
            "[Agent's Summary Description]",
            summary_description,
            "[Identity Anchor - highest priority]",
            _build_reflection_anchor(input.profile, input.retrieved_memories),
            f"It is {input.current_time.isoformat()}.",
            f"[{input.agent_identity.name}]'s status: {agent_status}.",
            f"Observation: {input.observation_content}",
        ]

        if input.dialogue_history:
            sections.append("Recent dialogue context:")
            for index, (partner_talk, my_talk) in enumerate(input.dialogue_history, start=1):
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
    default_trace = ReactionDecisionTrace(
        raw_response=response_text,
        parse_success=False,
        parse_error="json_parse_error_or_non_object",
        fallback_reason="parse_failure",
    )
    default_value = ReactionDecision(
        should_react=False,
        reaction="",
        reason="fallback",
        thought="",
        critique="",
        trace=default_trace,
    )
    parsed_json = _parse_json_object(response_text)
    repaired_once = False
    if parsed_json is None:
        repaired_payload = _attempt_json_repair_once(response_text)
        if repaired_payload is not None:
            parsed_json = repaired_payload
            repaired_once = True
        else:
            return default_value

    raw_should_react = parsed_json.get("should_react")
    if not isinstance(raw_should_react, bool):
        return replace(
            default_value,
            trace=replace(default_trace, parse_error="missing_or_invalid_should_react"),
        )

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
        trace=ReactionDecisionTrace(
            raw_response=response_text,
            parse_success=True,
            parse_error="repaired_once" if repaired_once else "",
        ),
    )


def _latest_partner_utterance(dialogue_history: list[tuple[str, str]]) -> str:
    for partner_talk, _ in reversed(dialogue_history):
        stripped = partner_talk.strip()
        if stripped and stripped != "none":
            return stripped
    return ""


def _parse_json_object(text: str) -> JsonObject | None:
    try:
        parsed = cast(object, json.loads(text))
    except json.JSONDecodeError:
        return None

    if not isinstance(parsed, dict):
        return None

    return cast(JsonObject, parsed)


def _attempt_json_repair_once(text: str) -> JsonObject | None:
    candidate = text.strip()
    if not candidate:
        return None

    first_open = candidate.find("{")
    if first_open > 0:
        candidate = candidate[first_open:]

    last_close = candidate.rfind("}")
    if last_close >= 0:
        candidate = candidate[: last_close + 1]

    open_count = candidate.count("{")
    close_count = candidate.count("}")
    if open_count > close_count:
        candidate = candidate + ("}" * (open_count - close_count))

    return _parse_json_object(candidate)


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
            "Lifestyle and routine: " + " | ".join(profile.extended.lifestyle_and_routine[:2])
        )

    if profile.extended.current_plan_context:
        summary_lines.append(
            "Current plan context: " + " | ".join(profile.extended.current_plan_context[:2])
        )

    return "\n".join(summary_lines)


def _build_agent_status(profile: AgentProfile) -> str:
    if profile.extended.current_plan_context:
        return profile.extended.current_plan_context[0]
    return "Idle"


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
            '- output json: {"should_react": true, "utterance": "그건 제 원칙과 맞지 않아서 도와드리기 어려워요.", "thought": "정체성과 충돌", "critique": "정중히 거절", "reason": "identity_conflict"}',
            "Example 2 (natural pivot to own interest):",
            "- input: partner asks a vague small-talk question during your focused routine",
            '- output json: {"should_react": true, "utterance": "짧게는 괜찮아요. 저는 요즘 디카프 추출 실험이 더 궁금해요.", "thought": "관심사로 전환", "critique": "과잉 협조 대신 자연스러운 화제 전환", "reason": "natural_topic_shift"}',
        ]
    )


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


def _recent_self_utterances(
    dialogue_history: list[tuple[str, str]],
    *,
    window: int,
) -> list[str]:
    if window < 1:
        return []

    self_utterances = [
        my_talk.strip()
        for _, my_talk in dialogue_history
        if my_talk and my_talk.strip() and my_talk.strip() != "none"
    ]
    return self_utterances[-window:]


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


def _max_ngram_overlap(candidate_sentence: str, recent_sentences: list[str]) -> float:
    if not recent_sentences:
        return 0.0
    return max(
        _overlap_ratio(candidate_sentence, recent_sentence, n=2)
        for recent_sentence in recent_sentences
    )


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


def _embed_sentences(
    *,
    sentences: list[str],
    embedding_encoder: "EmbeddingEncoder | None",
):
    if embedding_encoder is None:
        return []

    pairs = []
    for sentence in sentences:
        embedding = embedding_encoder.encode(EmbeddingEncodingContext(text=sentence))
        pairs.append((sentence, embedding))
    return pairs


@dataclass(frozen=True)
class SemanticOverlapCheck:
    max_similarity: float
    trigger: str


def _semantic_overlap_check(
    *,
    candidate_sentence: str,
    reference_sentences: list[str],
    reference_embeddings,
    embedding_encoder: "EmbeddingEncoder | None",
) -> SemanticOverlapCheck:
    if not candidate_sentence.strip() or not reference_sentences:
        return SemanticOverlapCheck(max_similarity=0.0, trigger="none")

    if embedding_encoder is None or not reference_embeddings:
        overlap = _max_ngram_overlap(candidate_sentence, reference_sentences)
        if overlap >= SEMANTIC_HARD_BLOCK_THRESHOLD:
            return SemanticOverlapCheck(max_similarity=overlap, trigger="hard")
        if overlap >= SEMANTIC_SOFT_PENALTY_THRESHOLD:
            return SemanticOverlapCheck(max_similarity=overlap, trigger="soft")
        return SemanticOverlapCheck(max_similarity=overlap, trigger="none")

    candidate_embedding = embedding_encoder.encode(
        EmbeddingEncodingContext(text=candidate_sentence)
    )

    best_similarity = 0.0
    for _, reference_embedding in reference_embeddings:
        similarity = float(cosine_similarity(candidate_embedding, reference_embedding))
        if similarity > best_similarity:
            best_similarity = similarity

    if best_similarity >= SEMANTIC_HARD_BLOCK_THRESHOLD:
        return SemanticOverlapCheck(max_similarity=best_similarity, trigger="hard")
    if best_similarity >= SEMANTIC_SOFT_PENALTY_THRESHOLD:
        return SemanticOverlapCheck(max_similarity=best_similarity, trigger="soft")
    return SemanticOverlapCheck(max_similarity=best_similarity, trigger="none")


def _build_semantic_guard_block(
    *,
    semantic_history: list[str],
    previous_candidate: str,
    max_similarity: float,
    trigger: str,
) -> str:
    level = "hard block" if trigger == "hard" else "soft penalty"
    lines = [
        f"Your previous reaction violated semantic repetition guard ({level}).",
        f"max_similarity={max_similarity:.3f}, soft={SEMANTIC_SOFT_PENALTY_THRESHOLD}, hard={SEMANTIC_HARD_BLOCK_THRESHOLD}",
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


def _build_partner_response_nudge_block(*, latest_partner_utterance: str) -> str:
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
class ReactionDecisionTrace:
    raw_response: str
    parse_success: bool
    parse_error: str = ""
    fallback_reason: str = ""
    suppress_reason: str = ""
    overlap_retry_count: int = 0
    partner_retry_count: int = 0
    semantic_retry_count: int = 0
    max_semantic_similarity: float = 0.0
    semantic_hard_threshold: float = SEMANTIC_HARD_BLOCK_THRESHOLD
    semantic_soft_threshold: float = SEMANTIC_SOFT_PENALTY_THRESHOLD
    semantic_retry_trigger: str = "none"


@dataclass(frozen=True)
class ReactionDecision:
    should_react: bool
    reaction: str
    reason: str
    thought: str = ""
    critique: str = ""
    trace: ReactionDecisionTrace = ReactionDecisionTrace(
        raw_response="",
        parse_success=False,
        parse_error="uninitialized",
    )


class GenerateClient(Protocol):
    def generate(
        self,
        *,
        prompt: str,
        system: str | None = None,
        options: OllamaGenerateOptions | None = None,
        format_json: bool = False,
    ) -> str: ...


class EmbeddingEncoder(Protocol):
    def encode(self, context: EmbeddingEncodingContext): ...
