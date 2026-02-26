import datetime
import re
from dataclasses import dataclass
from pathlib import Path

from .agent import AgentIdentity
from .agent_brain import AgentBrain


class SeedLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeedMemory:
    content: str
    importance: int


@dataclass(frozen=True)
class AgentSeed:
    agent: AgentIdentity
    identity_stable_set: list[str]
    lifestyle_and_routine: list[str]
    current_plan_context: list[str]
    seed_memories: list[SeedMemory]


class SeedLoader:
    def __init__(self, seed_dir: str | Path) -> None:
        self.seed_dir: Path = Path(seed_dir)

    def load(self, seed_name: str) -> AgentSeed:
        seed_path = self.seed_dir / f"{seed_name}.md"
        if not seed_path.exists():
            raise SeedLoadError(f"Seed file not found: {seed_path}")
        return self._parse_seed(seed_path.read_text(encoding="utf-8"))

    def load_all(self) -> list[AgentSeed]:
        seed_files = sorted(self.seed_dir.glob("*.md"))
        return [
            self._parse_seed(path.read_text(encoding="utf-8")) for path in seed_files
        ]

    @staticmethod
    def _parse_seed(text: str) -> AgentSeed:
        lines = text.splitlines()
        front_matter, body_lines = _split_front_matter(lines)
        agent = _parse_agent(front_matter)

        sections = _parse_sections(body_lines)
        identity_stable_set = _parse_bullets(sections.get("Identity Stable Set", []))
        lifestyle_and_routine = _parse_bullets(
            sections.get("Lifestyle and Routine", [])
        )
        current_plan_context = _parse_bullets(sections.get("Current Plan Context", []))
        seed_memories = _parse_seed_memories(sections.get("Seed Memories", []))

        return AgentSeed(
            agent=agent,
            identity_stable_set=identity_stable_set,
            lifestyle_and_routine=lifestyle_and_routine,
            current_plan_context=current_plan_context,
            seed_memories=seed_memories,
        )


def apply_seed_to_brain(
    *,
    brain: AgentBrain,
    seed: AgentSeed,
    now: datetime.datetime,
) -> None:
    for memory in seed.seed_memories:
        brain.save_observation_memory(
            content=memory.content,
            now=now,
            persona=seed.agent.name,
            importance=memory.importance,
        )


def _split_front_matter(lines: list[str]) -> tuple[list[str], list[str]]:
    if len(lines) < 3 or lines[0].strip() != "---":
        raise SeedLoadError("Seed front matter is missing")

    closing_index = None
    for index in range(1, len(lines)):
        if lines[index].strip() == "---":
            closing_index = index
            break

    if closing_index is None:
        raise SeedLoadError("Seed front matter is not closed")

    front_matter = lines[1:closing_index]
    body_lines = lines[closing_index + 1 :]
    return front_matter, body_lines


def _parse_agent(front_matter: list[str]) -> AgentIdentity:
    values: dict[str, str] = {}
    traits: list[str] = []
    current_key = ""

    for raw_line in front_matter:
        line = raw_line.rstrip()
        if not line:
            continue

        if line.startswith("  - ") or line.startswith("- "):
            if current_key != "traits":
                raise SeedLoadError("Traits list is malformed")
            traits.append(line.split("-", 1)[1].strip())
            continue

        if ":" not in line:
            raise SeedLoadError(f"Invalid front matter line: {line}")

        key, value = line.split(":", 1)
        current_key = key.strip()
        values[current_key] = value.strip()

    if "agent_id" not in values or "name" not in values or "age" not in values:
        raise SeedLoadError("Seed front matter requires agent_id, name, and age")

    return AgentIdentity(
        id=values["agent_id"],
        name=values["name"],
        age=int(values["age"]),
        traits=traits,
    )


def _parse_sections(body_lines: list[str]) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section = ""

    for raw_line in body_lines:
        line = raw_line.rstrip()
        if line.startswith("## "):
            current_section = line[3:].strip()
            sections[current_section] = []
            continue

        if current_section:
            sections[current_section].append(line)

    return sections


def _parse_bullets(lines: list[str]) -> list[str]:
    return [line[2:].strip() for line in lines if line.startswith("- ")]


def _parse_seed_memories(lines: list[str]) -> list[SeedMemory]:
    memory_pattern = re.compile(r"^\d+\.\s*\[importance:\s*(\d+)\]\s*(.+)$")
    seed_memories: list[SeedMemory] = []

    for line in lines:
        matched = memory_pattern.match(line)
        if matched is None:
            continue
        importance = int(matched.group(1))
        content = matched.group(2).strip()
        if content:
            seed_memories.append(SeedMemory(content=content, importance=importance))

    return seed_memories
