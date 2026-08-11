# SLM Orchestrator

`slm_orchestrator` is a lightweight, local semantic routing orchestrator powered entirely by a Small Language Model (SLM) running on CPU. It enables you to route user prompts dynamically to a custom list of agents with strict structured output constraints.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Robust Semantic Routing**: Utilizes dynamic few-shot prompt mapping and a 5-tier fallback parser to reliably map inputs to agents.
- **Agentic Tool Use**: Optional ReAct loop support. Pass custom tools (like Vector DB search) for the orchestrator to execute autonomously before routing.
- **Resource Efficient**: Uses a 1.5B parameter model (`Qwen 2.5 1.5B Instruct ONNX`), consuming only **1.5 GB to 2.0 GB of RAM** and taking **1.1 GB of disk storage**.
- **Highly Configurable**: Perfect for multi-agent systems, intent classification, and fallback routing.

---

## Installation

Install directly via `pip`:

```bash
pip install slm_orchestrator
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

### Tool Use Example

```python
from slm_orchestrator import SLMOrchestrator

orchestrator = SLMOrchestrator()

agents = [
    {"name": "RAG Agent", "description": "Searches and retrieves information from a knowledge base."},
    {"name": "Code Agent", "description": "Writes and executes Python code."},
    {"name": "General Agent", "description": "Handles general questions and conversations."}
]

tools = [
    {
        "name": "get_user_intent",
        "description": "Fetches more context about the user's intent from a database.",
        "parameters": {"query": "string"}
    }
]

def my_tool_executor(tool_name, args):
    if tool_name == "get_user_intent":
        return f"User intent context: {args.get('query')}"
    return "Unknown tool"

selected = orchestrator.route(
    agents=agents,
    question="Find all Python files that import pandas",
    tools=tools,
    tool_executor=my_tool_executor
)
print(f"Selected: {selected}")
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
    agents: list,                   # List of agent dicts with 'name' and 'description'
    question: str,                  # User query / question
    tools: list = None,             # Optional JSON schemas for tool use
    tool_executor: callable = None, # Optional callback function to execute tools
    max_iterations: int = 5         # Max ReAct tool execution loops
) -> str                            # Returns the exact name of the selected agent
```

**Agent dict format:**
```python
{
    "name": "Agent Name",           # str — must be unique
    "description": "What it does."  # str — used for routing decision
}
```

---

## Environment Variables

All constructor parameters can be overridden via environment variables:

| Variable | Description | Default |
|---|---|---|
| `SLM_ORCHESTRATOR_CONFIG` | Path to a custom `config.yaml` file | — |
| `SLM_ORCHESTRATOR_CACHE_DIR` | Override model download/cache directory | — |
| `SLM_ORCHESTRATOR_N_THREADS` | Number of CPU threads | `4` |
| `SLM_ORCHESTRATOR_N_CTX` | Context window size | `2048` |

---

## License

MIT License.
