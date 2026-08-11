# SLM Summarizer

`slm_summarizer` is a lightweight, local text summarization library powered entirely by a Small Language Model (SLM) running on CPU via ONNX Runtime. It allows developers to summarize short or large documents locally with high privacy, low resource usage, and zero API costs.

For longer documents that exceed typical memory/compute profiles, it dynamically applies a recursive Map-Reduce chunking pipeline, executing summarized chunks sequentially on local CPU without crashing or memory stutters.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Resource Efficient**: Uses a 1.5B parameter ONNX model (`Qwen 2.5 1.5B Instruct ONNX`), consuming only **1.5 GB to 2.0 GB of RAM** and taking **1.1 GB of disk storage**.
- **Map-Reduce for Large Text**: Automatically chunks long documents, summarizes each chunk independently, and recursively merges them into a final high-quality summary.
- **Evaluator-Corrector Loop**: Self-reflects on its own summaries. If a summary misses key points or fails the instruction, it automatically critiques and corrects itself.
- **Multiple Output Formats**: Supports bullet points (`bullet_points`), cohesive paragraphs (`paragraph`), and single-sentence TL;DRs (`tldr`).
- **Streaming Support**: Stream token-by-token output in real-time via a Python generator.
- **JSON Input API**: Accepts structured JSON input for easy integration with pipelines.

---

## Installation

Install directly via `pip`:

```bash
pip install slm-summarizer
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
from slm_summarizer import SLMSummarizer

# Initialize the summarizer (auto-locates or downloads the model)
summarizer = SLMSummarizer()

# Document to summarize
text = (
    "SpaceX successfully launched its Falcon 9 rocket on Friday, sending 22 Starlink satellites "
    "into low Earth orbit. The mission lifted off from Cape Canaveral Space Force Station in Florida. "
    "About eight minutes after launch, the rocket's first stage returned to Earth, landing safely on the "
    "droneship 'A Shortfall of Gravitas' stationed in the Atlantic Ocean. This marked the 15th successful "
    "flight and landing for this particular booster, representing another milestone in SpaceX's reuse technology. "
    "The Starlink constellation now provides high-speed satellite internet service to over 3 million subscribers globally."
)

# 1. Bullet point summary
bullet_summary = summarizer.summarize(text, format="bullet_points")
print("Bullets:\n", bullet_summary)

# 2. Single sentence TL;DR with custom styling instruction
tldr_summary = summarizer.summarize(
    text,
    format="tldr",
    instruction="Write in the style of a 17th-century pirate."
)
print("TL;DR:\n", tldr_summary)
```

### Streaming Example

```python
from slm_summarizer import SLMSummarizer

summarizer = SLMSummarizer()

text = "..."  # your document

for token in summarizer.summarize(text, format="paragraph", stream=True):
    print(token, end="", flush=True)
print()
```

### JSON Input Example

```python
from slm_summarizer import SLMSummarizer

summarizer = SLMSummarizer()

result = summarizer.summarize_json({
    "passage": "SpaceX launched 22 Starlink satellites...",
    "prompt": "Focus on the reusability milestone.",
    "size": 128,
    "format": "bullet_points"
})
print(result)
```

---

## Configuration API

```python
SLMSummarizer(
    model_path=None,   # Explicit path to an ONNX model directory (optional)
    cache_dir=None,    # Cache directory for auto-downloads
    n_ctx=8192,        # Context window size (default: 8192)
    n_threads=4        # Number of CPU threads (default: 4)
)
```

### Summarization API

```python
summarizer.summarize(
    text: str,                      # Document text to summarize
    format: str = "bullet_points",  # Output format: 'bullet_points', 'paragraph', or 'tldr'
    max_length: int = 256,          # Max token count for the final output
    instruction: str = "",          # Custom style or focus constraints
    chunk_size: int = 4000,         # Max character size per chunk for Map-Reduce chunking
    temperature: float = 0.0,       # 0.0 for deterministic summaries
    max_correction_loops: int = 0,  # Evaluator-corrector iterations (0 = disabled, fastest)
    stream: bool = False            # If True, returns a generator that yields token strings
)
```

### JSON Input API

```python
summarizer.summarize_json(
    json_input,                     # JSON string or dict
    format: str = "bullet_points",  # Fallback format if not set in json_input
    **kwargs                        # Additional kwargs forwarded to summarize()
) -> str
```

Accepted JSON keys:

| Key | Aliases | Description |
|---|---|---|
| `passage` | `text` | The text to summarize (required) |
| `prompt` | `instruction` | Style or focus constraint |
| `size` | `max_length` | Target max token count |
| `format` | `type` | Output format: `bullet_points`, `paragraph`, `tldr` |

---

## Environment Variables

All constructor parameters can be overridden via environment variables:

| Variable | Description | Default |
|---|---|---|
| `SLM_SUMMARIZER_CONFIG` | Path to a custom `config.yaml` file | — |
| `SLM_SUMMARIZER_CACHE_DIR` | Override model download/cache directory | — |
| `SLM_SUMMARIZER_N_THREADS` | Number of CPU threads | `4` |
| `SLM_SUMMARIZER_N_CTX` | Context window size | `8192` |
| `SLM_SUMMARIZER_MAX_LENGTH` | Default max output tokens | `256` |
| `SLM_SUMMARIZER_MAX_CORRECTION_LOOPS` | Default evaluator loop count | `0` |

---

## License

Apache License 2.0.
