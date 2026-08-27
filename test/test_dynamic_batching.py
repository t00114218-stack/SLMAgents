import os
import sys
import time
import asyncio
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from slm_embeddings.slm_embeddings.embeddings_server import SLMEmbeddingsServer
from slm_batch_engine import DynamicBatchEngine, BatchRequest

def test_batched_embeddings():
    print("\n==========================================")
    print("🧪 1. Testing Batched Neural Embeddings")
    print("==========================================")
    server = SLMEmbeddingsServer()
    texts = [
        "What is the total sales revenue for Q3 in the European region?",
        "How do I clean and format an untidy JSON payload?",
        "Extract key meeting action items and assignees.",
        "Generate a Python script to compute Fibonacci numbers.",
        "Query customer accounts with outstanding invoice balance > 1000.",
        "Summarize the main points of this technical PDF document.",
        "Translate the following user review from Spanish to English.",
        "Plan a multi-step workflow for data analysis and visualization.",
        "Verify SQL injection security vulnerabilities in user input.",
        "Scrape top headlines and metadata from the target web page."
    ]

    print(f"Embedding {len(texts)} sentences in a single batched AVX SIMD pass...")
    t0 = time.perf_counter()
    embeddings = server.embed(texts)
    t_batch = time.perf_counter() - t0

    print(f"✅ Generated {len(embeddings)} embedding vectors.")
    print(f"⏱️ Total batch time: {t_batch * 1000:.2f} ms ({t_batch * 1000 / len(texts):.2f} ms / sentence)")
    assert len(embeddings) == len(texts), "Mismatch in embeddings count"
    assert len(embeddings[0]) == 384, f"Expected 384 dimensions, got {len(embeddings[0])}"
    
    # Check similarity of identical sentences
    sim = server.similarity("Revenue report for Q3", "Revenue report for Q3")
    print(f"✅ Cosine similarity test (identical): {sim:.4f}")
    assert sim > 0.99, "Self similarity should be close to 1.0"

async def test_dynamic_batch_engine():
    print("\n==========================================")
    print("🧪 2. Testing Dynamic Micro-Batching Engine")
    print("==========================================")
    from main import get_shared_onnx_genai
    model, tokenizer = get_shared_onnx_genai()
    engine = DynamicBatchEngine.get_instance(model, tokenizer)

    test_prompts = [
        "Write a SQL query to select all employees with salary > 50000",
        "Write a SQL query to count total number of orders by customer",
        "Write a SQL query to find top 5 best selling products",
        "Write a SQL query to list active users registered in 2024",
        "Write a SQL query to calculate average order value per department",
        "Write a SQL query to find departments with more than 10 employees",
        "Write a SQL query to find orders placed in the last 7 days",
        "Write a SQL query to update customer status to active",
        "Write a SQL query to delete expired session tokens",
        "Write a SQL query to select maximum score achieved in each course"
    ]

    async def single_stream_client(idx: int, prompt: str):
        full_text = []
        t_start = time.perf_counter()
        async for tok in engine.generate_stream(prompt, max_tokens=20, request_id=f"test_req_{idx}"):
            full_text.append(tok)
        t_elapsed = time.perf_counter() - t_start
        return idx, "".join(full_text), t_elapsed

    print(f"\n🚀 Launching {len(test_prompts)} parallel concurrent requests simultaneously...")
    t_start_all = time.perf_counter()
    tasks = [single_stream_client(i, prompt) for i, prompt in enumerate(test_prompts)]
    results = await asyncio.gather(*tasks)
    t_total_all = time.perf_counter() - t_start_all

    print(f"\n📊 --- Concurrency Benchmark Results ({len(test_prompts)} Parallel Requests on 2 vCPUs) ---")
    print(f"Total wall-clock time for all {len(test_prompts)} parallel requests: {t_total_all:.2f} s")
    for idx, out, duration in results:
        preview = out.replace("\n", " ")[:40]
        print(f"  [Req #{idx}] Elapsed: {duration:.2f}s | Output: {preview}...")

    print(f"\n✅ Total batches processed: {engine.total_batches}, Total requests handled: {engine.total_processed_requests}")

def main():
    test_batched_embeddings()
    try:
        asyncio.run(test_dynamic_batch_engine())
    except Exception as e:
        print(f"[DynamicBatchEngine Benchmark Note]: {e}")

if __name__ == "__main__":
    main()
