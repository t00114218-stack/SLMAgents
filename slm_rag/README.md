# SLM RAG

`slm_rag` is a lightweight, local Retrieval-Augmented Generation (RAG) library powered entirely by a Small Language Model (SLM) running on CPU. It allows developers to pass a list of document chunks, a user question, and arbitrary guidelines/instructions to answer queries locally with high privacy, low resource usage, and zero API costs.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Resource Efficient**: Uses a 1.5B parameter model (`Qwen 2.5 1.5B Instruct ONNX`), consuming only **1.5 GB to 2.0 GB of RAM** and taking **1.1 GB of disk storage**.
- **Instruction Adherence**: Formats instructions directly into the system template to enforce constraints (e.g. style, safety, or formatting constraints like JSON).
- **Agentic Tool Use**: Optional ReAct loop support. Pass custom tools (like Vector DB search) for the RAG agent to autonomously fetch missing context before answering.

---

## Installation

Install directly via `pip`:

```bash
pip install slm-rag
```

Or install locally for development:

```bash
# 1. Create a fresh virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install the package in editable mode
pip install -e .
```

*Note: Requires `onnxruntime-genai`, `huggingface_hub`, and `pyyaml`.*

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
    model_path=None,   # Explicit path to an ONNX model directory (optional)
    cache_dir=None,    # Cache directory for auto-downloads
    n_ctx=8192,        # Context window size (default: 8192)
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
    max_tokens: int = 512,    # Maximum token limit for the response
    tools: list = None,       # Optional JSON schemas for tool use
    tool_executor: callable = None, # Optional callback function to execute tools
    max_iterations: int = 5   # Max ReAct tool execution loops
)
```
