"""
Pinecone 연결 및 레이턴시 벤치마크 테스트

Run with: uv run pytest tests/test_pinecone.py -v -s
"""

import os
import time

import pytest
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

load_dotenv()


@pytest.fixture(scope="module")
def pinecone_index():
    """Pinecone 인덱스 연결 fixture"""
    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "agent-crossing")

    if not api_key or api_key == "your_api_key_here":
        pytest.skip("PINECONE_API_KEY가 설정되지 않음")

    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    return pc.Index(index_name)


@pytest.mark.skip(reason="외부 서비스 필요 - 수동 실행: uv run pytest tests/test_pinecone.py -v -s")
class TestPinecone:
    """Pinecone 연결 및 레이턴시 테스트"""

    def test_upsert_latency(self, pinecone_index):
        """Upsert 레이턴시가 합리적인지 확인"""
        dummy_vector = [0.1] * 384

        start = time.time()
        pinecone_index.upsert(
            vectors=[
                {
                    "id": "test-1",
                    "values": dummy_vector,
                    "metadata": {"text": "Hello Pinecone"},
                }
            ]
        )
        upsert_time = time.time() - start

        assert upsert_time < 5.0, f"Upsert 레이턴시가 너무 높음: {upsert_time:.4f}s"

    def test_query_latency(self, pinecone_index):
        """Query 레이턴시가 합리적인지 확인"""
        dummy_vector = [0.1] * 384

        start = time.time()
        result = pinecone_index.query(vector=dummy_vector, top_k=1, include_metadata=True)
        query_time = time.time() - start

        assert query_time < 2.0, f"Query 레이턴시가 너무 높음: {query_time:.4f}s"
        assert "matches" in result, "Query 결과에 matches가 없음"
