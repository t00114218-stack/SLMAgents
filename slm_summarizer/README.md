# SLM Summarizer

`slm_summarizer` is a lightweight, local text summarization library powered entirely by a Small Language Model (SLM) running on CPU. It allows developers to summarize short or large documents locally with high privacy, low resource usage, and zero API costs. 

For longer documents that exceed typical memory/compute profiles, it dynamically applies a recursive Map-Reduce chunking pipeline, executing summarized chunks sequentially on local CPU without crashing or memory stutters.

---

## Key Features

- **Local & Private**: Runs completely on CPU / RAM. Zero API keys, zero network latency, and complete data privacy.
- **Resource Efficient**: Uses a 1.5B parameter model (`qwen2.5-1.5b-instruct-q4_k_m.gguf`), consuming only **1.0 GB to 1.5 GB of RAM**.
- **Map-Reduce for Large Text**: Automatically chunks long documents, summarizes each chunk independently, and recursively merges them into a final high-quality summary.
- **Evaluator-Corrector Loop**: Self-reflects on its own summaries. If a summary misses key points or fails the instruction, it automatically critiques and corrects itself.
- **Multiple Output Formats**: Supports bullet points (`bullet_points`), cohesive paragraphs (`paragraph`), and single-sentence TL;DRs (`tldr`).

---

## Installation

Install directly via `pip`:

```bash
pip install slm-summarizer
```

Or install locally for development:

```bash
# 1. Create a fresh virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install the package in editable mode
pip install -e .
```

*Note: Requires `llama-cpp-python`, `huggingface_hub`, and `pyyaml`.*

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

---

## Configuration API

```python
SLMSummarizer(
    model_path=None,   # Explicit path to a .gguf file (optional)
    cache_dir=None,    # Cache directory for auto-downloads (defaults to ~/.cache/slm_summarizer)
    n_ctx=8192,        # Context window size (default: 8192)
    n_threads=4        # Number of CPU threads (default: 4)
)
```

### Summarization API

```python
summarizer.summarize(
    text: str,                 # Document text to summarize
    format: str = "bullet_points", # Output format: 'bullet_points', 'paragraph', or 'tldr'
    max_length: int = 256,     # Max token count for the final output
    instruction: str = "",     # Custom style or focus constraints
    chunk_size: int = 4000,    # Max character size per chunk for Map-Reduce chunking
    temperature: float = 0.0,  # 0.0 for deterministic summaries
    max_correction_loops: int = 1 # Max evaluator-corrector iterations (default 1)
)
```
