# SLM Orchestrator 🧠🚀

`slm_orchestrator` is a lightweight, local semantic routing orchestrator powered entirely by a Small Language Model (SLM) running on CPU. It enables you to route user prompts dynamically to a custom list of agents with strict structured output constraints.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Dynamic Grammar Routing**: Automatically constructs a GBNF (GGML Backus-Naur Form) grammar on-the-fly to constrain the model's output to *exactly* one of the agent names you provide.
- **Transparent Model Download**: Scans your local directory or automatically downloads and caches the required 1B parameter instruct model on first execution.
- **Highly Configurable**: Perfect for multi-agent systems, intent classification, and fallback routing.

---

## Installation

Install directly via `pip`:

```bash
pip install slm-orchestrator
```

*Note: Requires `llama-cpp-python` and `huggingface_hub`.*

---

## Quick Start

```python
from slm_orchestrator import SLMOrchestrator

# Initialize the orchestrator (auto-downloads/loads the model)
orchestrator = SLMOrchestrator()

# Define your list of agents
agents = [
    {
        "name": "Billing Support", 
        "description": "Handles payments, invoices, refunds, and subscriptions."
    },
    {
        "name": "Technical Support", 
        "description": "Handles software installation, bug reports, and system crashes."
    },
    {
        "name": "General Chat", 
        "description": "Handles greetings, casual conversations, and general questions."
    }
]

# Route query
selected_agent = orchestrator.route(
    agents=agents,
    question="I need help with my monthly invoice payment"
)

print(f"Selected: {selected_agent}") 
# Output: Billing Support
```

---

## Configuration API

```python
SLMOrchestrator(
    model_path=None,   # Explicit path to a .gguf file (optional)
    cache_dir=None,    # Where to download the model (defaults to ~/.cache/slm_orchestrator)
    n_ctx=1024,        # Context size (default: 1024)
    n_threads=4        # CPU threads to run model generation (default: 4)
)
```

---

## License

MIT License.
