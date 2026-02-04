import os
from typing import List, Dict, Any, Optional
from memory.agent_memory import AgentMemory


class AgentBrain:
    """에이전트의 인지 및 의사결정을 담당하는 핵심 클래스"""

    def __init__(self, agent_id: str, name: str, persona: str):
        self.agent_id = agent_id
        self.name = name
        self.persona = persona
        self.memory = AgentMemory(index_name=f"agent-{agent_id}")

        # LLM 로드 (MLX 최적화 버전 우선 시도)
        try:
            from mlx_lm import load

            model_name = os.getenv(
                "LLM_MODEL_NAME", "mlx-community/Qwen2.5-3B-Instruct-4bit"
            )
            self.model, self.tokenizer = load(model_name)
            self.use_mlx = True
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_name = "Qwen/Qwen2.5-1.5B-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.use_mlx = False

    def perceive(self, observation: str, metadata: Optional[Dict[str, Any]] = None):
        """환경으로부터의 자극을 기억에 저장합니다."""
        self.memory.add_memory(observation, metadata)

    def decide(self, context: str) -> str:
        """현재 상황과 기억을 바탕으로 행동을 결정합니다."""
        # 관련 기억 회상
        relevant_memories = self.memory.retrieve_memories(context, top_k=3)
        memory_text = "\n".join([m.get("text", "") for m in relevant_memories])

        prompt = f"""
당신은 {self.name}입니다. {self.persona}
최근 기억:
{memory_text}

현재 상황: {context}
당신은 무엇을 할 것인가요? 짧게 답변하세요.
"""
        return self._generate(prompt)

    def converse(self, other_name: str, message: str) -> str:
        """타 NPC 또는 플레이어와의 대화를 생성합니다."""
        relevant_memories = self.memory.retrieve_memories(message, top_k=3)
        memory_text = "\n".join([m.get("text", "") for m in relevant_memories])

        prompt = f"""
당신은 {self.name}입니다. {self.persona}
최근 기억:
{memory_text}

{other_name}: {message}
{self.name}: 
"""
        return self._generate(prompt)

    def _generate(self, prompt: str) -> str:
        if self.use_mlx:
            from mlx_lm import generate

            messages = [{"role": "user", "content": prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            return generate(
                self.model, self.tokenizer, prompt=formatted_prompt, max_tokens=100
            )
        else:
            import torch

            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_new_tokens=100)
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt) :].strip()
