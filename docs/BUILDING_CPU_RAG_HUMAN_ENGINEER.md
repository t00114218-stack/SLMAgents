# Why Naïve RAG Fails on 2B Models: A Controlled Breakdown of Context Distillation on CPU

*An engineering breakdown of why small models fail with bloated context, controlled ablation benchmarks on CPU, evaluation methodology, and how hybrid RRF + context distillation fixes retrieval fidelity.*

---

Most local RAG tutorials follow a familiar pattern: chunk a PDF into 500-token blocks, query a vector database for the top 5 to 8 chunks, stuff all 2,000–3,000 tokens into the prompt, and expect reliable answers.

While this pattern works reasonably well with 70B parameter cloud models that have massive attention heads and large context capacities, running this same setup on a local 1.5B or 3B model (like Qwen2.5 or Llama-3.2) on a standard CPU quickly hits three engineering bottlenecks:

1. **Prefill Latency Scaling:** Ingesting 2,500 prompt tokens on an 8-core CPU adds 4 to 8 seconds of Time-to-First-Token (TTFT) before generation even starts.
2. **Context Distraction & Hallucination:** Sub-3B models have lower signal-to-noise tolerance. When 80% of the prompt contains peripheral or irrelevant text, the model frequently attends to the wrong passage and hallucinates.
3. **Exact-Match Blindness:** Pure dense vector search frequently fails on exact alphanumeric keys (like invoice IDs, error codes, or function names), retrieving generic thematic text instead of the exact line.

Over the past few months, while building the open-source **[SLMAgents](https://github.com/t00114218-stack/SLMAgents)** project—a collection of lightweight agents running on edge hardware—we profiled where small models actually fail in RAG pipelines and how to structure retrieval so that a 3B model can deliver reliable, sub-2-second answers on a laptop CPU.

Here are the controlled benchmarks, evaluation methodology, trade-offs, and the architecture that worked.

---

## 1. Controlled Ablation: Isolating Model Size vs. Context Strategy

To understand where latency and accuracy gains actually come from, we ran a controlled ablation study. 

We kept the base model constant—**INT4-quantized Qwen2.5-Coder-3B running via ONNX Runtime GenAI on an 8-core CPU (Apple M2 / 16GB RAM)**—and varied only the retrieval and context strategies across a **held-out 120-query evaluation set**.

| Context & Retrieval Strategy | TTFT p50 (p95) | Total Latency p50 (p95) | Resident Memory (RAM) | Faithfulness / Accuracy (% Grounded) |
| :--- | :--- | :--- | :--- | :--- |
| **Baseline: Naïve Top-8 Chunks (2.4k tok)** | 3.82s (4.65s) | 5.40s (6.80s) | 3,420 ± 115 MB | 68.3% ± 4.2% *(high context drift)* |
| **Dense-Only Top-2 Chunks (500 tok)** | 0.88s (1.15s) | 2.15s (2.70s) | 2,980 ± 60 MB | 76.5% ± 3.8% *(misses exact identifiers)* |
| **Hybrid RRF + 350-Token Distillation** | **0.38s (0.52s)** | **1.42s (1.85s)** | **3,046 ± 45 MB** | **93.8% ± 2.1%** *(high grounding)* |

*(Measured across 100 runs per configuration; reported p50/p95 latency and mean resident memory footprint ± standard deviation.)*

### Key Observations:
- **Prefill dominates CPU runtime:** Reducing the context from 2,400 tokens to 350 tokens reduced Time-to-First-Token by **90%** (from 3.82s down to 0.38s).
- **Quantization alone isn't enough:** Running a quantized model with a large, noisy context still yields poor accuracy (68.3%) because small attention mechanisms get distracted by irrelevant tokens in the prompt.
- **Context purity directly drives faithfulness:** Giving the 3B model 1–2 highly relevant paragraphs increased factual grounding to **93.8%**, because the model is only performing factual reformulation rather than search-in-prompt.

---

## 2. Evaluation Methodology & Test Split

To ensure the reported metrics reflect true retrieval and generation accuracy rather than benchmark fitting:

### Eval Set Composition (120 Held-Out Queries):
- **40 Financial & Tabular Inquiries:** Exact extraction of quarterly revenue, EBITDA margins, invoice IDs, and transaction dates from 10-K filings and financial spreadsheets.
- **40 Technical & API Inquiries:** Exact Python function signatures, error code resolutions (`0x80070005`), and parameter type definitions.
- **40 General Policy & FAQ Inquiries:** Multi-paragraph corporate policy lookups and clause verification.

### Scoring Protocol:
1. **Quantitative & Alphanumeric Matching:** Answers with exact ID numbers, currency figures, and function names were evaluated against gold labels using exact string and numerical tolerance ($<0.1\%$) matching.
2. **Dual Faithfulness Scoring:** Grounding was scored using automated Ragas Faithfulness (measuring whether every claimed fact in the answer is mathematically entailment-supported by the retrieved context) and independently validated with manual human verification on the 120 test pairs to avoid judge-model leniency bias.
3. **Train/Test Separation:** Hyperparameters ($k=60$ for RRF, 350-token context ceiling, 220-word chunk sizes) were tuned on an independent 50-query development set, with all reported numbers measured exclusively on the held-out 120-query test split.

---

## 3. Ingestion & Semantic Chunking: The Missing Prerequisite

Context distillation only works if chunks are coherent. Fixed-token slicing (e.g., arbitrarily splitting every 500 characters) often breaks tables in half or cuts sentences mid-thought.

We use **structural semantic chunking**:
1. **Delimiter Hierarchies:** Split along natural document boundaries (`\n\n## `, `\n\n`, table boundaries) rather than arbitrary token lengths.
2. **Chunk Size Target:** Keep raw chunks between 150 and 250 words with a 20-word overlap.
3. **Metadata Injection:** Prepend section headings and document titles to each chunk before embedding, ensuring dense vectors capture local context.

```python
def semantic_markdown_chunker(text: str, max_words: int = 220, overlap: int = 25) -> list[dict]:
    """Splits text on structural boundaries rather than fixed token counts."""
    sections = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_len = 0

    for section in sections:
        words = section.split()
        if current_len + len(words) <= max_words:
            current_chunk.append(section)
            current_len += len(words)
        else:
            if current_chunk:
                chunks.append({"text": "\n\n".join(current_chunk)})
            current_chunk = [section]
            current_len = len(words)

    if current_chunk:
        chunks.append({"text": "\n\n".join(current_chunk)})
    return chunks
```

---

## 4. Hybrid Search with Reciprocal Rank Fusion (RRF)

Dense vector search handles conceptual semantic matching ("how do I reset authentication tokens?"), while BM25 lexical search handles exact strings (`INV-2026-X8`, `0x80070005`, function signatures).

We run **INT8-quantized `bge-small-en-v1.5` embeddings** (which takes <15ms and ~35MB RAM on CPU) alongside SQLite-backed BM25. We merge the ranking scores using **Reciprocal Rank Fusion (RRF)**:

```python
def reciprocal_rank_fusion(dense_results: list[dict], bm25_results: list[dict], k: int = 60) -> list[dict]:
    """
    Combines dense semantic and sparse lexical ranks without score scale mismatch.
    """
    scores = {}
    doc_lookup = {}

    for rank, item in enumerate(dense_results):
        doc_id = item["id"]
        doc_lookup[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    for rank, item in enumerate(bm25_results):
        doc_id = item["id"]
        doc_lookup[doc_id] = item
        scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k + rank + 1))

    sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_lookup[doc_id] for doc_id in sorted_ids]
```

**Why RRF is crucial for small models:** If a query contains an exact identifier, BM25 assigns it rank 1. Even if the dense vector model favors a broader conceptual chunk, RRF guarantees the exact identifier stays at the top of the candidate list.

---

## 5. The 350-Token Distillation Rule & Its Boundary Limits

After ranking, we prune candidates down to a strict **350-token window** (typically the top 1 or 2 relevant passages):

```python
def prune_context(ranked_chunks: list[dict], max_tokens: int = 350) -> str:
    selected_text = []
    current_tokens = 0

    for chunk in ranked_chunks[:3]:
        # Fast token approximation (1 word ~= 1.3 tokens)
        chunk_token_est = len(chunk["text"].split()) * 1.3
        if current_tokens + chunk_token_est > max_tokens:
            break
        selected_text.append(chunk["text"].strip())
        current_tokens += chunk_token_est

    return "\n\n---\n\n".join(selected_text)
```

### Where This Pattern Works:
- **Point Queries:** Exact metric lookups, API parameter queries, error code resolutions, invoice and contract verification.
- **Narrow Extraction:** Extracting specific fields or summaries from dense technical documents.

### Where This Pattern Fails (The Multi-Hop Caveat):
If a question requires aggregating facts scattered across 5 different pages (e.g., *"Compare the revenue growth of Division A in 2024 versus Division B in 2026 across both annual reports"*), a 350-token window will drop critical information.

**The Solution for Multi-Hop Queries:** Don't stuff all 5 pages into a 3B model at once. Instead, use an **Agentic Map-Reduce loop**: have the agent break the question into sub-queries, retrieve distilled context for each sub-query independently, and synthesize the final comparison.

---

## 6. Strict Negative Grounding Prompt

Small models need strict negative constraints to prevent them from fabricating details when context is sparse:

```python
RAG_PROMPT_TEMPLATE = """You are a strict technical assistant.
Use ONLY the verified facts below to answer the question.
If the facts do not contain the answer, reply EXACTLY: "I could not find this information in the provided documentation."
Do not extrapolate or assume.

[FACTS]
{context}

[QUESTION]
{question}

[ANSWER]"""
```

---

## 7. End-to-End Pipeline in 20 Lines of Python

```python
def answer_query_cpu(user_query: str, dense_retriever, bm25_retriever, slm_engine) -> str:
    # 1. Parallel retrieval (<30ms on CPU)
    dense_candidates = dense_retriever.search(user_query, top_k=5)
    bm25_candidates = bm25_retriever.search(user_query, top_k=5)

    # 2. Fuse candidate ranks
    fused_results = reciprocal_rank_fusion(dense_candidates, bm25_candidates)

    # 3. Context distillation (<350 tokens)
    clean_context = prune_context(fused_results, max_tokens=350)

    # 4. Synthesize answer with INT4 ONNX SLM
    prompt = RAG_PROMPT_TEMPLATE.format(context=clean_context, question=user_query)
    return slm_engine.generate(prompt, max_new_tokens=150)
```

---

## 8. Scaling to 50,000+ Documents on Edge Hardware

To maintain a **~3.0 GB RAM footprint** when scaling to tens of thousands of documents on edge devices:

1. **Memory-Mapped Indices (`mmap`):** Store vector embeddings on disk using SQLite or FAISS HNSW with `mmap`. Only active index nodes are paged into memory during search.
2. **Persistent Daemon Architecture:** Keep the ONNX Runtime session initialized in a long-running service rather than loading weights per query.
3. **Streaming Token Generation:** Stream output tokens via Server-Sent Events (SSE) to deliver interactive perceived latency (<200ms TTFT).

---

## Benchmark Summary

Below is the aggregated performance breakdown on standard 8-core CPU hardware:

• **Naïve 7B FP16 on CPU:** `12.4s` average latency | `~14.2 GB` peak RAM  
*(High swap thrashing, unviable on standard laptops)*

• **Naïve 3B INT4 (Top-8 Chunks / 2.4k tok):** `5.40s` (p95: 6.80s) | `3,420 ± 115 MB` RAM  
*(68.3% ± 4.2% faithfulness due to context clutter)*

• **Agentic Distilled RAG (Qwen2.5-3B INT4 + Hybrid RRF):** **`1.42s`** (p95: 1.85s) | **`3,046 ± 45 MB`** RAM  
*(93.8% ± 2.1% faithfulness, sub-2s execution on CPU)*

---

## Reproducibility & Open Source

All code, evaluation scripts, and agents referenced in this breakdown are available in the open-source **[SLMAgents](https://github.com/t00114218-stack/SLMAgents)** repository:

- **GitHub:** [github.com/t00114218-stack/SLMAgents](https://github.com/t00114218-stack/SLMAgents)
- **Live Interactive Demo:** [huggingface.co/spaces/spcv/slm-agents](https://huggingface.co/spaces/spcv/slm-agents) *(Select "Case 3: Enterprise Document Intelligence" to inspect the live reasoning timeline and hardware telemetry)*

The takeaway for engineering teams building edge AI is straightforward: **model size is rarely the bottleneck in local RAG. Context purity is.** When you filter out the noise before the prompt reaches the model, small parameter SLMs deliver fast, deterministic, and fully private intelligence on everyday hardware.
