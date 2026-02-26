import argparse
import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from agents.agent import AgentIdentity
from agents.agent_brain import AgentBrain
from agents.reflection import Reflection
from agents.reflection_service import ReflectionService
from agents.seed_loader import SeedLoader, apply_seed_to_brain
from llm.embedding_encoder import OllamaEmbeddingEncoder
from llm.importance_scorer import OllamaImportanceScorer
from llm.llm_service import LlmService
from llm.ollama_client import OllamaClient
from memory.memory_service import MemoryService
from memory.memory_stream import MemoryStream


@dataclass
class SimAgent:
    agent: AgentIdentity
    brain: AgentBrain
    memory_service: MemoryService

    @property
    def name(self) -> str:
        return self.agent.name


def build_agent(
    agent: AgentIdentity, ollama_client: OllamaClient, llm_model: str
) -> SimAgent:
    memory_stream = MemoryStream()
    importance_scorer = OllamaImportanceScorer(client=ollama_client, model=llm_model)
    embedding_encoder = OllamaEmbeddingEncoder(client=ollama_client)
    memory_service = MemoryService(
        memory_stream=memory_stream,
        importance_scorer=importance_scorer,
        embedding_encoder=embedding_encoder,
    )
    llm_service = LlmService(ollama_client)
    reflection_service = ReflectionService(
        reflection=Reflection(),
        memory_service=memory_service,
        llm_service=llm_service,
    )
    brain = AgentBrain(
        memory_service=memory_service,
        reflection_service=reflection_service,
    )
    return SimAgent(
        agent=agent,
        brain=brain,
        memory_service=memory_service,
    )


def build_reply_prompt(
    speaker: str,
    partner: str,
    history: list[tuple[str, str]],
    turn_index: int,
    language: str,
) -> str:
    history_lines = [f"{name}: {message}" for name, message in history[-8:]]
    history_text = "\n".join(history_lines) if history_lines else "No prior dialogue."
    language_instruction = (
        "Respond only in Korean (Hangul)."
        if language == "ko"
        else "Respond only in English."
    )
    return (
        f"You are {speaker}. You are having a casual conversation with {partner}.\n"
        f"Turn number: {turn_index}.\n"
        "Keep your response to one short sentence.\n"
        f"{language_instruction}\n"
        "No markdown, no quotes, no role prefix.\n\n"
        f"Conversation so far:\n{history_text}\n\n"
        f"Respond as {speaker}:"
    )


def is_allowed_text(text: str, language: str) -> bool:
    for char in text:
        if not char.isalpha():
            continue

        codepoint = ord(char)
        is_basic_latin = (0x0041 <= codepoint <= 0x005A) or (
            0x0061 <= codepoint <= 0x007A
        )
        is_hangul = (
            0x1100 <= codepoint <= 0x11FF
            or 0x3130 <= codepoint <= 0x318F
            or 0xA960 <= codepoint <= 0xA97F
            or 0xAC00 <= codepoint <= 0xD7A3
            or 0xD7B0 <= codepoint <= 0xD7FF
        )

        if language == "ko" and not is_hangul:
            return False

        if language == "en" and not is_basic_latin:
            return False

    return True


def generate_reply_with_language_guard(
    *,
    ollama_client: OllamaClient,
    llm_model: str,
    base_prompt: str,
    language: str,
) -> str:
    prompt = base_prompt
    for _ in range(3):
        reply = ollama_client.generate(model=llm_model, prompt=prompt).strip()
        if is_allowed_text(reply, language):
            return reply

        rewrite_instruction = (
            "Rewrite in one short sentence using only Korean (Hangul)."
            if language == "ko"
            else "Rewrite in one short sentence using only English."
        )
        prompt = (
            f"{base_prompt}\n\n"
            "Your previous answer used disallowed scripts. "
            f"{rewrite_instruction}"
        )

    return "좋아요. 그렇게 할게요." if language == "ko" else "Sounds good."


def ingest_line(observer: SimAgent, content: str, now: datetime.datetime) -> None:
    observer.brain.queue_observation(
        content=content,
        now=now,
        persona=observer.name,
    )


def run_simulation(
    *,
    agent_a_seed_name: str,
    agent_b_seed_name: str,
    turns: int,
    base_url: str,
    llm_model: str,
    timeout_seconds: float,
    seed_dir: str,
    language: str,
) -> None:
    ollama_client = OllamaClient(base_url=base_url, timeout_seconds=timeout_seconds)
    seed_loader = SeedLoader(seed_dir)
    seed_a = seed_loader.load(agent_a_seed_name)
    seed_b = seed_loader.load(agent_b_seed_name)

    agent_a = build_agent(seed_a.agent, ollama_client, llm_model)
    agent_b = build_agent(seed_b.agent, ollama_client, llm_model)

    bootstrap_time = datetime.datetime.now()
    apply_seed_to_brain(brain=agent_a.brain, seed=seed_a, now=bootstrap_time)
    apply_seed_to_brain(brain=agent_b.brain, seed=seed_b, now=bootstrap_time)

    history: list[tuple[str, str]] = []

    for turn in range(1, turns + 1):
        speaker = agent_a if turn % 2 == 1 else agent_b
        listener = agent_b if speaker is agent_a else agent_a

        prompt = build_reply_prompt(
            speaker=speaker.name,
            partner=listener.name,
            history=history,
            turn_index=turn,
            language=language,
        )
        reply = generate_reply_with_language_guard(
            ollama_client=ollama_client,
            llm_model=llm_model,
            base_prompt=prompt,
            language=language,
        )
        history.append((speaker.name, reply))
        print(f"[{turn:02d}] {speaker.name}: {reply}")

        now = datetime.datetime.now()
        ingest_line(listener, f"{speaker.name} said: {reply}", now)
        ingest_line(speaker, f"I said to {listener.name}: {reply}", now)
        speaker.brain.action_loop()
        listener.brain.action_loop()

    print("\nRecent memories")
    for agent in (agent_a, agent_b):
        print(f"\n- {agent.name}")
        memories = agent.memory_service.get_recent_memories(limit=5)
        for memory in memories:
            print(f"  [{memory.node_type.value}] {memory.content}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    default_seed_dir = str(Path(__file__).resolve().parents[1] / "seeds")
    _ = parser.add_argument("--agent-a", default="Jiho")
    _ = parser.add_argument("--agent-b", default="Sujin")
    _ = parser.add_argument("--turns", type=int, default=20)
    _ = parser.add_argument("--base-url", default="http://localhost:11434")
    _ = parser.add_argument("--llm-model", default="qwen2.5:7b-instruct")
    _ = parser.add_argument("--timeout-seconds", type=float, default=30.0)
    _ = parser.add_argument("--seed-dir", default=default_seed_dir)
    _ = parser.add_argument("--language", choices=["ko", "en"], default="ko")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    agent_a_seed_name = cast(str, args.agent_a)
    agent_b_seed_name = cast(str, args.agent_b)
    turns = cast(int, args.turns)
    base_url = cast(str, args.base_url)
    llm_model = cast(str, args.llm_model)
    timeout_seconds = cast(float, args.timeout_seconds)
    seed_dir = cast(str, args.seed_dir)
    language = cast(str, args.language)

    run_simulation(
        agent_a_seed_name=agent_a_seed_name,
        agent_b_seed_name=agent_b_seed_name,
        turns=max(1, turns),
        base_url=base_url,
        llm_model=llm_model,
        timeout_seconds=timeout_seconds,
        seed_dir=seed_dir,
        language=language,
    )


if __name__ == "__main__":
    main()
