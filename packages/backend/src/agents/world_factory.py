from agents.agent import AgentContext, AgentProfile
from agents.agent_brain import AgentBrain
from agents.reflection import Reflection
from agents.reflection_service import ReflectionService
from agents.seed_loader import AgentSeed
from agents.sim_agent import SimAgent
from llm.embedding_encoder import OllamaEmbeddingEncoder
from llm.importance_scorer import OllamaImportanceScorer
from llm.llm_service import LlmService
from llm.ollama_client import OllamaClient
from memory.memory_service import MemoryService
from memory.memory_stream import MemoryStream


def _profile_from_seed(seed: AgentSeed) -> AgentProfile:
    return AgentProfile(
        identity_stable_set=list(seed.identity_stable_set),
        lifestyle_and_routine=list(seed.lifestyle_and_routine),
        current_plan_context=list(seed.current_plan_context),
    )


def build_agent(
    seed: AgentSeed, ollama_client: OllamaClient, llm_model: str
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
        agent_identity=seed.agent,
        memory_service=memory_service,
        reflection_service=reflection_service,
    )
    context = AgentContext(
        identity=seed.agent,
        profile=_profile_from_seed(seed),
        brain=brain,
        memory_service=memory_service,
    )
    return SimAgent(context=context)
