import json
from dataclasses import dataclass
from typing import cast

from agents.memory.memory_object import MemoryObject
from llm.ollama_client import JsonObject, OllamaClient


class LlmService:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client: OllamaClient = ollama_client

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


@dataclass(frozen=True)
class InsightWithCitation:
    context: str
    citation_memory_ids: list[int]
