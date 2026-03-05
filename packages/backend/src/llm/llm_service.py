import json
from dataclasses import dataclass
from typing import cast

from agents.memory.memory_object import MemoryObject
from llm.guardrails.similarity import EmbeddingEncoder
from llm.governance import (
    DayPlanParseError,
    GenerateClient,
    ReactionDecision,
    ReactionDecisionInput,
    ReactionPipeline,
    try_parse_day_plan_broad_strokes,
)
from llm.ollama_client import JsonObject, OllamaGenerateOptions

from . import prompt_builders


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
        self.reaction_pipeline: ReactionPipeline = ReactionPipeline(
            ollama_client=ollama_client,
            embedding_encoder=embedding_encoder,
        )

    def generate_salient_high_level_questions(
        self, agent_name: str, memories: list[MemoryObject]
    ) -> list[str]:
        if not memories:
            return []

        prompt = prompt_builders.build_salient_questions_prompt(
            agent_name=agent_name,
            memories=memories,
        )
        response_text = self.ollama_client.generate(prompt=prompt, format_json=True)

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
        return self.reaction_pipeline.decide_reaction(
            input,
            generation_options=generation_options,
        )
