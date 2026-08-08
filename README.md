# SLM Agents 🧠⚡

Welcome to the **SLM Agents** community! This repository is a unified developer portal and codebase for running highly-constrained, secure, and privacy-first agentic workflows locally on standard CPUs.

No GPU rigs, no subscription costs, and zero network latency.

---

## Repository Structure

This monorepo is organized into the following main projects:

| Folder | Description | Installation |
| :--- | :--- | :--- |
| [**`slm_orchestrator`**](./slm_orchestrator) | Semantic router powered by 1B Small Language Models using GBNF grammar constraints. | `pip install slm-orchestrator` |
| [**`slm_rag`**](./slm_rag) | High-efficiency local CPU Retrieval-Augmented Generation library (under 1.5 GB RAM). | `pip install slm-rag` |
| [**`website`**](./website) | The developer landing page, portal, and community website. | (See below) |

---

## Getting Started

### 🧠 SLM Orchestrator
Route user prompts dynamically to specialized agents with strict GBNF output constraints.
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

### 🔍 SLM RAG
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
print(answer) # Ahoy matey! AegisShield be their flagship!
```

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
