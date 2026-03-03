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
from .reaction_parsing import parse_reaction_decision
from .reaction_types import (
    GenerateClient,
    ReactionDecision,
    ReactionDecisionInput,
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

    def decide_reaction(
        self,
        input: ReactionDecisionInput,
        *,
        generation_options: OllamaGenerateOptions | None = None,
    ) -> ReactionDecision:
        prompt = prompt_builders.build_reaction_decision_prompt(
            agent_identity=input.agent_identity,
            current_time=input.current_time,
            observation_content=input.observation_content,
            dialogue_history=input.dialogue_history,
            profile=input.profile,
            retrieved_memories=input.retrieved_memories,
        )
        system_prompt = prompt_builders.language_system_prompt(input.language)
        recent_sentences = recent_dialogue_sentences(input.dialogue_history, window=3)

        retry_count = 0
        max_retry = 2
        semantic_retry_count = 0
        max_semantic_retry = 2
        partner_retry_count = 0
        working_prompt = prompt
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
            decision = parse_reaction_decision(response_text)

            if (
                partner_utterance
                and not decision.should_react
                and partner_retry_count < 1
            ):
                partner_retry_count += 1
                working_prompt = (
                    f"{prompt}\n\n"
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
                working_prompt = f"{prompt}\n\n{semantic_guard}"
                continue

            if (
                not decision.should_react
                or not decision.reaction
                or not recent_sentences
            ):
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
            working_prompt = f"{prompt}\n\n{overlap_guard}"
