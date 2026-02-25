from dataclasses import dataclass


@dataclass
class Agent:
    """에이전트의 기본 정보와 특성을 담는 클래스."""

    id: str  # 고유 id
    name: str  # 에이전트 이름
    age: int  # 에이전트 나이
    traits: list[str]  # 성격 특성 (예: 친절함, 호기심 등)
