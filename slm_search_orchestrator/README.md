# SLM Search Orchestrator

A lightweight, local CPU-optimized query expansion and search reranking agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It generates optimized search strings, retrieves results offline using the **duckduckgo-search** library, and structures snippets.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Search Orchestrator acts as a pre-retrieval query planner to improve accuracy in Retrieval-Augmented Generation (RAG) pipelines.

```
+------------------+
| User Query Input |
+--------+---------+
         |
         v
+------------------------------------------+
| Query Expansion (Phi-3.5)                |
| (Generate exactly 3 search variations)   |
+--------+---------+
         |
         v
+------------------------------------------+
| Search Execution (DuckDuckGo DDGS)       |
+--------+---------+
         |
         v
+------------------------------------------+
| Snippet De-duplication & Filtering       |
| (Remove duplicate URLs and text noise)   |
+--------+---------+
         |
         v
+------------------+
| Consolidated     |
| Context Snippets |
+------------------+
```

### Retrieval & Orchestration Flow:
1. **Query Expansion:** Small models often fail to retrieve relevant documents because the initial user query lacks specific terminology. The agent queries Phi-3.5 to generate exactly 3 variations of the search query (addressing acronyms, synonym alternatives, and technical variants).
2. **Offline Snippet Fetching:** The agent executes queries via the `duckduckgo-search` API, retrieving titles, body summaries, and link URLs.
3. **De-duplication & Clean-up:** Aggregates all result lists, filters out duplicate link paths, and trims down redundant sentences to fit within context limits.
4. **Mock Fallback Handling:** If the network is offline or the search API returns rate limits, the orchestrator falls back to mock database lookups to guarantee stability.

---

## ⚡ 2. CPU Performance Tuning Guidelines

Running concurrent searches and text processing can bottleneck local CPU performance. Follow these recommendations:

1. **Limit Search Breadth:**
   * Restrict search parameters (`max_results_per_query=2` or `3`). Fetching too many pages increases token overhead, slowing down downstream RAG processing.
2. **Stateful Prompt Loading:**
   * Keep the orchestrator model loaded in memory between runs rather than instantiating the class repeatedly, which reduces model loading overhead.
3. **Optimal Threading Configuration:**
   * Align thread execution with standard environment variables:
     ```python
     orchestrator = SLMSearchOrchestrator(n_threads=4)
     ```

---

## 🎯 3. Accuracy Optimization Tips

*   **Query Count Enforcement:** Small models can occasionally generate long conversational responses when asked to output query variations. Use strict instructions:
    *   *Instruct:* "Output exactly a valid JSON array of 3 strings: `[\"query1\", \"query2\", \"query3\"]`. Return no other conversational text."
*   **Prompt Boundary Alignment:** Use correct Phi-3.5 chat format tags:
    ```text
    <|system|>
    You are an offline query expansion assistant.
    Return only a valid JSON list containing exactly 3 query variations.<|end|>
    <|user|>
    Query: {user_query}<|end|>
    <|assistant|>
    ```
*   **Mock Verification:** Ensure fallback paths return structured keys matching the schema so that your pipeline doesn't crash when internet access is unavailable.

---

## 📂 4. API Reference

### `SLMSearchOrchestrator`

```python
from slm_search_orchestrator.search_orchestrator import SLMSearchOrchestrator

orchestrator = SLMSearchOrchestrator(
    model_path="../../models/phi-3.5-mini-instruct-onnx",
    n_ctx=4096,
    n_threads=4
)
```

#### Methods

##### `generate_queries(user_query: str) -> list[str]`
Expands the raw search goal into exactly 3 query variations.
* **`user_query`** (*str*): The input user query.
* **Returns**: *list[str]* containing query strings.

##### `retrieve(user_query: str, max_results_per_query: int = 2) -> list[dict]`
Executes queries via DuckDuckGo and compiles the results.
* **`user_query`** (*str*): The input user query.
* **`max_results_per_query`** (*int*): Limit of results per query variation.
* **Returns**: *list[dict]* containing `title`, `href` (URL), and `body` (snippet content).

---

## 🚀 5. Usage Example

Here is a realistic usage example demonstrating query planning, expansion, and execution to gather technical information:

```python
from slm_search_orchestrator.search_orchestrator import SLMSearchOrchestrator

orchestrator = SLMSearchOrchestrator()

# Execute retrieval
result = orchestrator.retrieve("CPU inference thread optimization settings")

print(result)
```

### Generated Output Response:
```json
[
  {
    "title": "Configuring OMP_NUM_THREADS for CPU Inference",
    "href": "https://docs.slmagents.ai/cpu-threads",
    "body": "For optimal ONNX CPU inference, set OMP_NUM_THREADS to match the physical core count, disabling hyperthreading overhead."
  },
  {
    "title": "Optimizing Local SLM Performance on CPU",
    "href": "https://blog.slmagents.ai/slm-cpu-tuning",
    "body": "Small Language Models run highly efficiently on CPU by mapping threads to core boundaries, keeping memory allocations flat."
  }
]
```
