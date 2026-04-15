from llm.guardrails.similarity import EmbeddingEncoder

from .contracts import GenerateClient, ReactionDecision, ReactionDecisionInput
from .reaction_graph import ReactionGraphRunner


class ReactionPipeline:
    def __init__(
        self,
        *,
        ollama_client: GenerateClient,
        embedding_encoder: EmbeddingEncoder | None,
    ):
        self.reaction_graph: ReactionGraphRunner = ReactionGraphRunner(
            ollama_client=ollama_client,
            embedding_encoder=embedding_encoder,
        )

    def decide_reaction(
        self,
        input: ReactionDecisionInput,
    ) -> ReactionDecision:
        return self.reaction_graph.decide_reaction(input)
