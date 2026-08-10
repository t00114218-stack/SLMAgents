# SLM Agents

Welcome to the **SLM Agents** community! This repository is a unified open-source developer portal and Python codebase for running highly-constrained, secure, and privacy-first AI agent workflows locally on standard CPUs using Small Language Models (SLMs). 

If you are looking for a **local AI framework**, **CPU inference library**, or **privacy-first LLM orchestrator**, you are in the right place! No GPU rigs, no subscription costs, and zero network latency. Powered by ONNX Runtime GenAI for maximum speed.

---

## Repository Structure

This monorepo is organized into the following main projects:

| Folder | Description | Installation |
| :--- | :--- | :--- |
| [**`slm_orchestrator`**](./slm_orchestrator) | Semantic router powered by 1.5B ONNX models with dynamic few-shot mapping and autonomous tool use. | `pip install slm-orchestrator` |
| [**`slm_rag`**](./slm_rag) | High-efficiency local CPU Retrieval-Augmented Generation library via ONNX Runtime with autonomous tool use. | `pip install slm-rag` |
| [**`slm_summarizer`**](./slm_summarizer) | High-efficiency local CPU text summarization agent via ONNX Runtime with an evaluator-corrector loop. | `pip install slm-summarizer` |
| [**`slm_text_to_sql`**](./slm_text_to_sql) | CPU-optimized Text-to-SQL agent via ONNX Runtime, with built-in QLoRA TPU/GPU fine-tuning. | `pip install slm-text-to-sql` |
| [**`website`**](./website) | The developer landing page, portal, and community website. | (See below) |

---

## Getting Started

### SLM Orchestrator
Route user prompts dynamically to specialized agents with robust semantic mapping constraints.
```python
from slm_orchestrator import SLMOrchestrator

orchestrator = SLMOrchestrator()
agents = [
    {"name": "Billing Support", "description": "Handles payment issues"},
    {"name": "General Chat", "description": "casual queries"}
]

selected = orchestrator.route(agents=agents, question="I need invoice help")
print(selected) # Output: Billing Support
```

### SLM RAG
Answer queries locally using context documents with strict guideline compliance.
```python
from slm_rag import SLMRag

rag = SLMRag()
chunks = ["NebulaCorp flagship product is AegisShield."]
answer = rag.answer(
    chunks=chunks,
    question="What is their flagship product?",
    instruction="Answer like a 17th-century pirate."
)
print(answer) # Output: Ahoy matey! AegisShield be their flagship!
```

### SLM Summarizer
Summarize short or large documents locally on standard CPUs using single-pass or Map-Reduce methods.
```python
from slm_summarizer import SLMSummarizer

summarizer = SLMSummarizer()
text = "SpaceX successfully launched its Falcon 9 rocket on Friday..."
summary = summarizer.summarize(
    text=text,
    format="bullet_points",
    instruction="Focus on launch metrics."
)
print(summary)
```

You can also pass inputs as a JSON string or dict mapping prompt and target summary size (number of tokens):
```python
import json

json_input = json.dumps({
    "passage": "SpaceX successfully launched its Falcon 9 rocket...",
    "prompt": "Focus on launch metrics",
    "size": 50,
    "format": "tldr"
})
summary = summarizer.summarize_json(json_input)
print(summary)
```

### SLM Text-to-SQL
Generate SQL queries from natural language database questions on standard CPUs.
```python
from slm_text_to_sql import SLMTextToSQL

agent = SLMTextToSQL()
schema = "CREATE TABLE employees (id INT, name VARCHAR(50), salary INT);"
question = "Get the names of employees earning more than 50000."
query = agent.generate_sql(schema=schema, question=question)
print(query) # Output: SELECT name FROM employees WHERE salary > 50000;
```



---

## Model Details & Hardware Requirements

All three libraries in this monorepo are optimized for local CPU execution via **ONNX Runtime GenAI** and share a single model cache to minimize resource overhead:

*   **Model**: `tonythethompson/Qwen2.5-1.5B-Instruct-ONNX` (INT4 quantized).
*   **License**: **Apache 2.0** (100% Permissive). Safe for commercial distribution without restrictive LLM community agreements.
*   **Memory Footprint (RAM)**: **~1.5 GB to 2.0 GB** during active inference.
*   **Disk Storage**: **~1.1 GB** total. The model directory is cached under `~/.cache/slm_summarizer/qwen2.5-1.5b-onnx` and shared across orchestrator, RAG, and summarizer libraries to prevent redundant downloads.

---

## Developer Website

The community landing page is built using HTML5 and Vanilla CSS. To run the developer portal locally:

1. Navigate to the website folder:
   ```bash
   cd website
   ```
2. Start a local server:
   ```bash
   python3 -m http.server 8000
   ```
3. Open your browser and navigate to:
   ```
   http://localhost:8000/index.html
   ```

---

## Contributing

We welcome contributions! Please feel free to open Issues or pull requests to improve the libraries or expand the roadmap for upcoming agents (such as SQL Agent, Code Interpreter, and Web Crawler).

Released under the **MIT License**.
