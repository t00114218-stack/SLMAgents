# SLM Search Orchestrator

A lightweight, local CPU-optimized query expansion and search reranking agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It generates optimized search strings, retrieves results offline using the **duckduckgo-search** library, and structures snippets.

---

## Features

- **MIT-Licensed & Permissive**: Exclusively uses MIT/Apache 2.0 components.
- **Search Query Expansion**: Generates multiple search keywords to maximize coverage.
- **Local Snippet Aggregator**: Structures scraped text blocks into unified contexts.
- **Offline Planning**: Evaluates and splits query inputs on device.

---

## Installation

```bash
pip install -e ./slm_search_orchestrator
```

---

## API Reference

### `SLMSearchOrchestrator`

```python
from slm_search_orchestrator.search_orchestrator import SLMSearchOrchestrator

orchestrator = SLMSearchOrchestrator()
```

#### `generate_queries(user_query: str) -> list[str]`
Expands the raw search goal into distinct query variations.

#### `retrieve(user_query: str, max_results_per_query: int = 2) -> list[dict]`
- **Arguments**:
  - `user_query` (str): Search text request.
  - `max_results_per_query` (int): Limit of results per query variation.
- **Returns**:
  - `list[dict]`: Stored dictionaries of structured snippet elements.
