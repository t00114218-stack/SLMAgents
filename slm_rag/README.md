# SLM RAG 🧠🔍

`slm_rag` is a lightweight, local Retrieval-Augmented Generation (RAG) library powered entirely by a Small Language Model (SLM) running on CPU. It allows developers to pass a list of document chunks, a user question, and arbitrary guidelines/instructions to answer queries locally with high privacy, low resource usage, and zero API costs.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Resource Efficient**: Uses a 1.5B parameter model (`qwen2.5-1.5b-instruct-q4_k_m.gguf`), consuming only **1.0 GB to 1.5 GB of RAM**.
- **Instruction Adherence**: Formats instructions directly into the system template to enforce constraints (e.g. style, safety, or formatting constraints like JSON).

---

## Installation

Install directly via `pip`:

```bash
pip install slm-rag
```

Or install locally for development:

```bash
# 1. Create a fresh virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install the package in editable mode
pip install -e .
```

*Note: Requires `llama-cpp-python` and `huggingface_hub`.*

---

## Quick Start

```python
from slm_rag import SLMRag

# Initialize the RAG engine (auto-locates or downloads the model)
rag = SLMRag()

# Provide context chunks
chunks = [
    "NebulaCorp was founded in 2024 by Dr. Helena Vance. It specializes in quantum-resistant encryption algorithms.",
    "The flagship product of NebulaCorp is called 'AegisShield'. It is widely used by financial organizations.",
    "In early 2026, NebulaCorp announced a partnership with the European Space Agency."
]

# Run query with a strict instruction
answer = rag.answer(
    chunks=chunks,
    question="What is their flagship product?",
    instruction="Answer like a 17th-century pirate.",
    temperature=0.0
)

print(answer)
# Output: "Ahoy matey! AegisShield be the flagship product of NebulaCorp, savvy?"
```

---

## Configuration API

```python
SLMRag(
    model_path=None,   # Explicit path to a .gguf file (optional)
    cache_dir=None,    # Cache directory for auto-downloads (defaults to ~/.cache/slm_rag)
    n_ctx=2048,        # Context window size (default: 2048)
    n_threads=4        # Number of CPU threads (default: 4)
)
```

### Answering Queries

```python
rag.answer(
    chunks: list[str],      # Document text chunks
    question: str,          # User query / question
    instruction: str,       # Instruction or constraint the model must follow
    temperature: float = 0.0, # Generation temperature (0.0 for deterministic answers)
    max_tokens: int = 512    # Maximum token limit for the response
)
```
