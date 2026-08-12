---
title: SLM Agents
emoji: 🎯
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
github_repo: t00114218-stack/SLMAgents
---

# SLM Agents

[![Open in Hugging Face](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/YOUR_HF_USERNAME/SLMAgents)

Welcome to the **SLM Agents** community! This repository is a unified open-source developer portal and Python codebase for running highly-constrained, secure, and privacy-first AI agent workflows locally on standard CPUs using Small Language Models (SLMs). 

If you are looking for a **local AI framework**, **CPU inference library**, or **privacy-first LLM orchestrator**, you are in the right place! No GPU rigs, no subscription costs, and zero network latency. Powered by ONNX Runtime GenAI for maximum speed.

---

## Repository Structure

This monorepo is organized into the following main projects:

| Folder | Description | Installation |
| :--- | :--- | :--- |
| [**`slm_orchestrator`**](./slm_orchestrator) | Semantic router powered by 1.5B ONNX models with dynamic few-shot mapping and autonomous routing. | `pip install slm-orchestrator` |
| [**`slm_rag`**](./slm_rag) | High-efficiency local CPU Retrieval-Augmented Generation library via ONNX Runtime with autonomous tool use. | `pip install slm-rag` |
| [**`slm_summarizer`**](./slm_summarizer) | High-efficiency local CPU text summarization agent via ONNX Runtime with an evaluator-corrector loop. | `pip install slm-summarizer` |
| [**`slm_text_to_sql`**](./slm_text_to_sql) | CPU-optimized Text-to-SQL agent via ONNX Runtime, with built-in QLoRA TPU/GPU fine-tuning. | `pip install slm-text-to-sql` |
| [**`slm_cli_agent`**](./slm_cli_agent) | CPU-optimized command line helper translating language to safe executable commands. | `pip install -e ./slm_cli_agent` |
| [**`slm_code_interpreter`**](./slm_code_interpreter) | Sandboxed Python execution context with auto self-correction traceback recovery loops. | `pip install -e ./slm_code_interpreter` |
| [**`slm_git_copilot`**](./slm_git_copilot) | Git diff helper translating raw diff streams to conventional commits. | `pip install -e ./slm_git_copilot` |
| [**`slm_json_cleaner`**](./slm_json_cleaner) | Pattern mapping utility repairing malformed/broken JSON strings to fit target schemas. | `pip install -e ./slm_json_cleaner` |
| [**`slm_document_parser`**](./slm_document_parser) | Structure-aware layout parser converting PDF/DOCX to markdown and extracting JSON data. | `pip install -e ./slm_document_parser` |
| [**`slm_vision_parser`**](./slm_vision_parser) | Visual layout parser and OCR processor for whiteboard flowcharts and image diagrams. | `pip install -e ./slm_vision_parser` |
| [**`slm_web_agent`**](./slm_web_agent) | Interactive browser controller executing clicks, navigation, and page actions. | `pip install -e ./slm_web_agent` |
| [**`slm_web_scraper`**](./slm_web_scraper) | Fast, offline HTML cleaning utility filtering noise and generating clean text blocks. | `pip install -e ./slm_web_scraper` |
| [**`slm_search_orchestrator`**](./slm_search_orchestrator) | Local search aggregator routing complex questions to web scrapers and synthesis engines. | `pip install -e ./slm_search_orchestrator` |
| [**`slm_db_migration`**](./slm_db_migration) | Legacy schema diffing agent generating migration DDLs verified in SQLite sandboxes. | `pip install -e ./slm_db_migration` |
| [**`slm_email`**](./slm_email) | Secure email inbox analyzer extracting action items and auto-replying with set tone. | `pip install -e ./slm_email` |
| [**`slm_meeting`**](./slm_meeting) | Conversation post-processor formatting action logs and decisions to markdown tables. | `pip install -e ./slm_meeting` |
| [**`slm_voice`**](./slm_voice) | Low-latency audio handler supporting spoken command transcriptions on local CPU. | `pip install -e ./slm_voice` |
| [**`slm_memory`**](./slm_memory) | SQLite-backed entity memory store mapping and retrieving user preferences. | `pip install -e ./slm_memory` |
| [**`slm_task_planner`**](./slm_task_planner) | Autonomous goal decomposition engine mapping dependency DAGs to specialized sub-agents. | `pip install -e ./slm_task_planner` |
| [**`slm_pdf`**](./slm_pdf) | Document conversation RAG interface built on top of the document parser modules. | `pip install -e ./slm_pdf` |
| [**`slm_pkb`**](./slm_pkb) | Markdown indexing agent generating semantic link relationships in personal knowledge bases. | `pip install -e ./slm_pkb` |
| [**`slm_data`**](./slm_data) | Offline CSV dataset analyst parsing profiles and executing pandas evaluation script tasks. | `pip install -e ./slm_data` |
| [**`slm_translation`**](./slm_translation) | Multilingual dynamic translation engine supporting local NLLB-200 and IndicTrans2 models. | `pip install -e ./slm_translation` |
| [**`slm_math`**](./slm_math) | Natural language arithmetic parser solving equations symbolically with SymPy. | `pip install -e ./slm_math` |
| [**`slm_security`**](./slm_security) | Security guardrail audit redacting PII data and blocking malicious command injections. | `pip install -e ./slm_security` |
| [**`slm_embeddings`**](./slm_embeddings) | Local vector embeddings server calculating cosine semantic similarity of text. | `pip install -e ./slm_embeddings` |


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

### Claude-Style Streaming & Reasoning
All three generation agents (`SLMSummarizer`, `SLMRag`, `SLMTextToSQL`) support Claude-style streaming. When `stream=True` is passed, the model will output its step-by-step thinking inside `<thought>...</thought>` tags first, followed by the final answer. The method returns a Python generator yielding decoded tokens in real-time.

```python
# Real-time streaming with thought-process parsing
stream = rag.answer(
    chunks=["NebulaCorp released AegisShield in 2025."],
    question="What is the release year of AegisShield?",
    instruction="State only the year.",
    stream=True
)

for token in stream:
    # Outputs the reasoning block first in <thought>...</thought> tags,
    # then the final clean answer.
    print(token, end="", flush=True)
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
