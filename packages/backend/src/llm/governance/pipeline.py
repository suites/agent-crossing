from dataclasses import replace

from llm import prompt_builders
from llm.guardrails.similarity import (
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
from llm.clients.ollama import OllamaGenerateOptions

from .contracts import (
    GenerateClient,
    ReactionDecision,
    ReactionDecisionInput,
    ReactionIntent,
    ReactionUtterance,
)
from .parsing import parse_reaction_intent, parse_reaction_utterance


class ReactionPipeline:
    def __init__(
        self,
        *,
        ollama_client: GenerateClient,
        embedding_encoder: EmbeddingEncoder | None,
    ):
        self.ollama_client: GenerateClient = ollama_client
        self.embedding_encoder: EmbeddingEncoder | None = embedding_encoder

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
