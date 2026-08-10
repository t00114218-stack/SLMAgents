# SLM Orchestrator

`slm_orchestrator` is a lightweight, local semantic routing orchestrator powered entirely by a Small Language Model (SLM) running on CPU. It enables you to route user prompts dynamically to a custom list of agents with strict structured output constraints.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Robust Semantic Routing**: Utilizes dynamic few-shot prompt mapping and multi-stage fallback parser to map inputs to agents.
- **Agentic Tool Use**: Optional ReAct loop support. Pass custom tools (like Vector DB search) for the orchestrator to execute autonomously before routing.
- **Resource Efficient**: Uses a 1.5B parameter model (`Qwen 2.5 1.5B Instruct ONNX`), consuming only **1.5 GB to 2.0 GB of RAM** and taking **1.1 GB of disk storage**.
- **Highly Configurable**: Perfect for multi-agent systems, intent classification, and fallback routing.

---

## Installation

Install directly via `pip`:

```bash
pip install slm-orchestrator
```

*Note: Requires `onnxruntime-genai`, `huggingface_hub`, and `pyyaml`.*

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
    model_path=None,   # Explicit path to an ONNX model directory (optional)
    cache_dir=None,    # Where to download the model
    n_ctx=2048,        # Context size (default: 2048)
    n_threads=4        # CPU threads (default: 4)
)
```

### Routing API

```python
orchestrator.route(
    agents: list,                 # List of agent dicts with 'name' and 'description'
    question: str,                # User query / question
    tools: list = None,           # Optional JSON schemas for tool use
    tool_executor: callable = None, # Optional callback function to execute tools
    max_iterations: int = 5       # Max ReAct tool execution loops
)
```

---

## License

MIT License.
