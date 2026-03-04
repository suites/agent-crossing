import json
from dataclasses import dataclass, replace
from typing import cast

from agents.memory.memory_object import MemoryObject
from llm.ollama_client import JsonObject, OllamaGenerateOptions

from . import prompt_builders
from .reaction_guards import (
    EmbeddingEncoder,
    SEMANTIC_HARD_BLOCK_THRESHOLD,
    SEMANTIC_SOFT_PENALTY_THRESHOLD,
    embed_sentences,
    exceeds_ngram_overlap_threshold,
    latest_partner_utterance,
    recent_dialogue_sentences,
    recent_self_utterances,
    semantic_overlap_check,
)
from .reaction_parsing import (
    DayPlanParseError,
    parse_reaction_intent,
    parse_reaction_utterance,
    try_parse_day_plan_broad_strokes,
)
from .reaction_types import (
    GenerateClient,
    ReactionDecision,
    ReactionDecisionInput,
    ReactionIntent,
    ReactionUtterance,
)


@dataclass(frozen=True)
class InsightWithCitation:
    context: str
    citation_memory_ids: list[int]


class LlmService:
    def __init__(
        self,
        ollama_client: GenerateClient,
        *,
        embedding_encoder: EmbeddingEncoder | None = None,
    ):
        self.ollama_client: GenerateClient = ollama_client
        self.embedding_encoder: EmbeddingEncoder | None = embedding_encoder

    def generate_salient_high_level_questions(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list[str]:
        if not memories:
            return []

        prompt = prompt_builders.build_salient_questions_prompt(
            agent_name=agent_name,
            memories=memories,
        )
        response_text = self.ollama_client.generate(prompt=prompt)

        try:
            parsed_data = cast(object, json.loads(response_text))
            if not isinstance(parsed_data, dict):
                return []

            parsed_payload = cast(JsonObject, parsed_data)
            questions = parsed_payload.get("questions", [])
            if not isinstance(questions, list):
                return []

            parsed_questions: list[str] = []
            for question in cast(list[object], questions):
                if isinstance(question, str) and question.strip():
                    parsed_questions.append(question)

            return parsed_questions
        except json.JSONDecodeError:
            return []

    def generate_insights_with_citation_key(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list[InsightWithCitation]:
        if not memories:
            return []

        statement_to_memory_id = {
            index + 1: memory.id for index, memory in enumerate(memories)
        }
        prompt = prompt_builders.build_insights_with_citation_prompt(
            agent_name=agent_name,
            memories=memories,
        )
        response_text = self.ollama_client.generate(prompt=prompt, format_json=True)

        try:
            parsed_data = cast(object, json.loads(response_text))
            if not isinstance(parsed_data, dict):
                return []

            parsed_payload = cast(JsonObject, parsed_data)
            insights = parsed_payload.get("insights", [])
            if not isinstance(insights, list):
                return []

            parsed_insights: list[InsightWithCitation] = []
            for raw_insight in cast(list[object], insights):
                if not isinstance(raw_insight, dict):
                    continue

                insight_payload = cast(JsonObject, raw_insight)
                insight_text = insight_payload.get("insight")
                citation_numbers = insight_payload.get("citation_statement_numbers")
                if not isinstance(insight_text, str) or not insight_text.strip():
                    continue

                citation_memory_ids: list[int] = []
                if isinstance(citation_numbers, list):
                    for raw_number in cast(list[object], citation_numbers):
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

    def generate_day_plan_broad_strokes(
        self,
        *,
        agent_name: str,
        age: int,
        innate_traits: list[str],
        persona_background: str,
        yesterday_date_text: str,
        yesterday_summary: str,
        today_date_text: str,
    ) -> list[str]:
        """Generate broad-strokes day plan with bounded retries on parse failures."""
        prompt = prompt_builders.build_day_plan_broad_strokes_prompt(
            agent_name=agent_name,
            age=age,
            innate_traits=innate_traits,
            persona_background=persona_background,
            yesterday_date_text=yesterday_date_text,
            yesterday_summary=yesterday_summary,
            today_date_text=today_date_text,
        )

        current_prompt = prompt
        max_parse_retries = 2
        for attempt in range(max_parse_retries + 1):
            response_text = self.ollama_client.generate(
                prompt=current_prompt,
                format_json=True,
            )

            try:
                return try_parse_day_plan_broad_strokes(response_text).broad_strokes
            except DayPlanParseError as exc:
                if attempt >= max_parse_retries:
                    return []

                current_prompt = self._build_day_plan_broad_strokes_retry_prompt(
                    base_prompt=prompt,
                    previous_error=exc.reason,
                    previous_response=response_text,
                )

        return []

    @staticmethod
    def _build_day_plan_broad_strokes_retry_prompt(
        *,
        base_prompt: str,
        previous_error: str,
        previous_response: str,
    ) -> str:
        """Build a stricter follow-up prompt when prior JSON output is invalid."""
        return (
            f"{base_prompt}\n\n"
            "The previous response did not match the required broad-strokes JSON schema.\n"
            f"Failure reason: {previous_error}.\n\n"
            'Return JSON only with this exact shape and no extra text: "{"broad_strokes": ["<stroke 1>", ...]}"\n'
            f"Do not repeat this invalid output: {previous_response[:180]!r}"
        )

    def decide_reaction(
        self,
        input: ReactionDecisionInput,
        *,
        generation_options: OllamaGenerateOptions | None = None,
    ) -> ReactionDecision:
        intent_prompt = prompt_builders.build_reaction_intent_prompt(
            agent_identity=input.agent_identity,
            current_time=input.current_time,
            observation_content=input.observation_content,
            dialogue_history=input.dialogue_history,
            profile=input.profile,
            retrieved_memories=input.retrieved_memories,
        )
        system_prompt = prompt_builders.language_system_prompt(input.language)

        intent_response = self.ollama_client.generate(
            prompt=intent_prompt,
            system=system_prompt,
            options=generation_options,
            format_json=True,
        )
        intent = parse_reaction_intent(intent_response)
        if not intent.should_react:
            return ReactionDecision(
                should_react=False,
                reaction="",
                reason=intent.reason,
                thought=intent.thought,
                critique=intent.critique,
                trace=replace(
                    intent.trace,
                    partner_retry_count=0,
                    semantic_hard_threshold=SEMANTIC_HARD_BLOCK_THRESHOLD,
                    semantic_soft_threshold=SEMANTIC_SOFT_PENALTY_THRESHOLD,
                ),
            )

        utterance_prompt = prompt_builders.build_reaction_utterance_prompt(
            agent_identity=input.agent_identity,
            current_time=input.current_time,
            observation_content=input.observation_content,
            dialogue_history=input.dialogue_history,
            profile=input.profile,
            retrieved_memories=input.retrieved_memories,
            intent_reason=intent.reason,
            intent_thought=intent.thought,
            intent_critique=intent.critique,
        )

        recent_sentences = recent_dialogue_sentences(input.dialogue_history, window=3)

        retry_count = 0
        max_retry = 2
        semantic_retry_count = 0
        max_semantic_retry = 2
        partner_retry_count = 0
        working_prompt = utterance_prompt
        partner_utterance = latest_partner_utterance(input.dialogue_history)

        semantic_history = recent_self_utterances(input.dialogue_history, window=5)
        semantic_history_embeddings = embed_sentences(
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
            utterance_result = parse_reaction_utterance(response_text)
            decision = self._build_reaction_decision(
                intent=intent,
                utterance_result=utterance_result,
            )

            if partner_utterance and not decision.reaction and partner_retry_count < 1:
                partner_retry_count += 1
                working_prompt = (
                    f"{utterance_prompt}\n\n"
                    + prompt_builders.build_partner_response_nudge_block(
                        latest_partner_utterance=partner_utterance
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

            semantic_check = semantic_overlap_check(
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

                semantic_guard = prompt_builders.build_semantic_guard_block(
                    semantic_history=semantic_history,
                    previous_candidate=decision.reaction,
                    max_similarity=semantic_check.max_similarity,
                    trigger=semantic_check.trigger,
                    soft_threshold=SEMANTIC_SOFT_PENALTY_THRESHOLD,
                    hard_threshold=SEMANTIC_HARD_BLOCK_THRESHOLD,
                )
                working_prompt = f"{utterance_prompt}\n\n{semantic_guard}"
                continue

            if not decision.reaction or not recent_sentences:
                return replace(
                    decision,
                    trace=replace(
                        decision.trace,
                        semantic_retry_count=semantic_retry_count,
                        max_semantic_similarity=semantic_check.max_similarity,
                        semantic_retry_trigger=semantic_check.trigger,
                    ),
                )

            has_overlap = exceeds_ngram_overlap_threshold(
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

            overlap_guard = prompt_builders.build_overlap_guard_block(
                recent_sentences=recent_sentences,
                previous_candidate=decision.reaction,
            )
            working_prompt = f"{utterance_prompt}\n\n{overlap_guard}"

    def _build_reaction_decision(
        self,
        *,
        intent: ReactionIntent,
        utterance_result: ReactionUtterance,
    ) -> ReactionDecision:
        return ReactionDecision(
            should_react=True,
            reaction=utterance_result.utterance,
            reason=(utterance_result.reason or intent.reason).strip() or "n/a",
            thought=(utterance_result.thought or intent.thought).strip(),
            critique=(utterance_result.critique or intent.critique).strip(),
            trace=utterance_result.trace,
        )
