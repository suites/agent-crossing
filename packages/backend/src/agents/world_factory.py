import datetime
from pathlib import Path

from agents.agent import AgentContext, AgentProfile, ExtendedPersona, FixedPersona
from agents.agent_brain import AgentBrain
from agents.persona_loader import AgentPersona, PersonaLoader, apply_persona_to_brain
from agents.reflection import Reflection
from agents.reflection_service import ReflectionService
from agents.sim_agent import SimAgent
from llm.embedding_encoder import OllamaEmbeddingEncoder
from llm.importance_scorer import OllamaImportanceScorer
from llm.llm_service import LlmService
from llm.ollama_client import OllamaClient

from .memory.memory_service import MemoryService
from .memory.memory_stream import MemoryStream


def _profile_from_persona(persona: AgentPersona) -> AgentProfile:
    return AgentProfile(
        fixed=FixedPersona(identity_stable_set=list(persona.identity_stable_set)),
        extended=ExtendedPersona(
            lifestyle_and_routine=list(persona.lifestyle_and_routine),
            current_plan_context=list(persona.current_plan_context),
        ),
    )


def build_agent(
    persona: AgentPersona, ollama_client: OllamaClient, llm_model: str
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
        agent_name=persona.agent.name,
        identity_stable_set=list(persona.identity_stable_set),
    )
    brain = AgentBrain(
        agent_identity=persona.agent,
        memory_service=memory_service,
        reflection_service=reflection_service,
        llm_service=llm_service,
    )
    context = AgentContext(
        identity=persona.agent,
        profile=_profile_from_persona(persona),
        brain=brain,
        memory_service=memory_service,
    )
    return SimAgent(context=context)


def init_agents(
    *,
    persona_dir: str | Path,
    agent_persona_names: list[str],
    ollama_client: OllamaClient,
    llm_model: str,
    now: datetime.datetime,
) -> list[SimAgent]:
    if not agent_persona_names:
        raise ValueError("agent_persona_names must not be empty")

    persona_loader = PersonaLoader(persona_dir)

    agents: list[SimAgent] = []
    for persona_name in agent_persona_names:
        persona = persona_loader.load(persona_name)
        agent = build_agent(persona, ollama_client, llm_model)
        apply_persona_to_brain(brain=agent.brain, persona=persona, now=now)
        agents.append(agent)

    return agents
