import os
import time
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec


def test_pinecone_setup():
    load_dotenv()

    api_key = os.getenv("PINECONE_API_KEY")
    index_name = os.getenv("PINECONE_INDEX_NAME", "agent-crossing")

    if not api_key or api_key == "your_api_key_here":
        print("❌ Error: PINECONE_API_KEY is not set in .env file.")
        return

    print(f"Connecting to Pinecone...")
    pc = Pinecone(api_key=api_key)

    if index_name not in pc.list_indexes().names():
        print(f"Index '{index_name}' not found. Creating it...")
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(index_name).status["ready"]:
            time.sleep(1)

    index = pc.Index(index_name)
    print(f"✅ Successfully connected to index: {index_name}")

    print("\nRunning simple latency benchmark...")
    dummy_vector = [0.1] * 384

    start = time.time()
    index.upsert(
        vectors=[
            {
                "id": "test-1",
                "values": dummy_vector,
                "metadata": {"text": "Hello Pinecone"},
            }
        ]
    )
    upsert_time = time.time() - start
    print(f"✓ Upsert latency: {upsert_time:.4f}s")

    start = time.time()
    index.query(vector=dummy_vector, top_k=1, include_metadata=True)
    query_time = time.time() - start
    print(f"✓ Query latency: {query_time:.4f}s")

    print("\nBenchmark Summary:")
    print(f"Average Roundtrip: {(upsert_time + query_time) / 2:.4f}s")


if __name__ == "__main__":
    test_pinecone_setup()
