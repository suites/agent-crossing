import json

from llm.ollama_client import OllamaClient
from memory.memory_object import MemoryObject


class LlmService:
    def __init__(self, ollama_client: OllamaClient):
        self.ollama_client = ollama_client

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
            parsed_data = json.loads(response_text)
            return parsed_data.get("questions", [])
        except json.JSONDecodeError:
            # 포맷이 깨졌을 경우의 방어 코드
            return []
