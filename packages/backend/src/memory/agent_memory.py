import os
import uuid
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()


class AgentMemory:
    def __init__(
        self,
        index_name: Optional[str] = None,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("PINECONE_API_KEY is not set in .env")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", "agent-crossing"
        )
        self.index = self.pc.Index(self.index_name)

        self.model = SentenceTransformer(embedding_model)

    def add_memory(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        memory_id = str(uuid.uuid4())
        embedding = self.model.encode(text).tolist()

        meta = metadata or {}
        meta["text"] = text

        self.index.upsert(
            vectors=[{"id": memory_id, "values": embedding, "metadata": meta}]
        )
        return memory_id

    def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.model.encode(query).tolist()

        results = self.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )

        return [match.metadata for match in results.matches if match.metadata]
