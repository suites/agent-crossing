from dataclasses import dataclass
from typing import cast

from agents.agent import AgentContext, AgentIdentity, AgentProfile
from agents.agent_brain import AgentBrain

from .memory.memory_service import MemoryService


@dataclass
class SimAgent:
    context: AgentContext

    @property
    def name(self) -> str:
        return self.context.identity.name

    @property
    def identity(self) -> AgentIdentity:
        return self.context.identity

    @property
    def profile(self) -> AgentProfile:
        return self.context.profile

    @property
    def brain(self) -> AgentBrain:
        return cast(AgentBrain, self.context.brain)

    @property
    def memory_service(self) -> MemoryService:
        return cast(MemoryService, self.context.memory_service)
