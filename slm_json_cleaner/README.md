# SLM JSON Cleaner

A lightweight, CPU-optimized local JSON text sanitizer and structural repair tool powered by a local Small Language Model (SLM) running via ONNX Runtime GenAI. It is built specifically to repair unstructured, malformed, cut-off, or poorly formatted text strings and clean them into valid JSON structures complying with a user-defined target schema.

---

## Features

- **Malformed JSON Repair**: Repairs broken strings, missing brackets, unquoted keys, trailing commas, and unclosed quotes.
- **Strict Schema Enforcement**: Forces the structured output keys and types to align exactly with a target dictionary schema.
- **Brace Matcher Fallback**: If markdown syntax is malformed, uses an inline regular expression brace-extraction fallback to guarantee parsing success.
- **Single-Pass Efficiency**: Avoids complex state-tracking pipelines; takes a single prompt instruction block to execute pattern mappings.

---

## Installation

In your local project environment:
```bash
pip install -e ./slm_json_cleaner
```

Ensure `onnxruntime-genai` is installed. It shares the central monorepo model path cached locally at `models/qwen2.5-1.5b-onnx`.

---

## API Reference

### `SLMJSONCleaner`

```python
from slm_json_cleaner.json_cleaner import SLMJSONCleaner

cleaner = SLMJSONCleaner(
    model_path=None,   # Path to the ONNX model directory (defaults to models/qwen2.5-1.5b-onnx)
    cache_dir=None,    # Alternative HF cache dir
    n_ctx=2048,        # Context length (defaults to 2048)
    n_threads=4        # Number of CPU threads to use for execution
)
```

## API Reference

### `SLMJSONCleaner`

```python
from slm_json_cleaner.json_cleaner import SLMJSONCleaner

cleaner = SLMJSONCleaner(
    model_path=None,   # Path to the ONNX model directory (defaults to models/qwen2.5-1.5b-onnx)
    cache_dir=None,    # Alternative HF cache dir
    n_ctx=2048,        # Context length (defaults to 2048)
    n_threads=4        # Number of CPU threads to use for execution
)
```

#### Methods

#### `clean_json(malformed_text: str, schema_dict: dict)`
Cleans broken or unstructured text to conform with `schema_dict`.
- **Arguments**:
  - `malformed_text` (str): Raw string containing malformed or cut-off JSON.
  - `schema_dict` (dict): A reference dictionary representing the expected keys and value types.
- **Returns**:
  - `tuple[dict, bool]`: `(parsed_json_dict, success_flag)`. Returns `{"raw_output": text, "error": msg}` and `False` if parsing fails completely.

---

## Usage Examples

### 1. Repairing Complex Truncated & Nested JSON
Here is a realistic scenario where an LLM's response was abruptly cut off mid-token due to context window limits, leaving unclosed brackets, missing keys, and dangling commas intermingled with conversational logs:

```python
from slm_json_cleaner.json_cleaner import SLMJSONCleaner

cleaner = SLMJSONCleaner()

# A highly malformed, cut-off raw string block:
malformed_input = """
Below is the output log matching build parameters:
{
  "project_name": "Antigravity Pipeline",
  "build_status": "success",
  "metrics": {
     "duration_seconds": 124.5,
     "test_count": 48,
     "failed_tests": 0,
     "coverage": "98.5%
  },
  "contributors": [
     {"name": "Alice", "role": "lead"},
     {"name": "Bob", "role": "reviewer"
  ],
  "releases": [
     "v1.0", "v1.1", 
  
  
[LOG EXPIRED - TOKEN LIMIT REACHED]
"""

# The target schema we want to enforce:
target_schema = {
    "project_name": "string",
    "build_status": "string",
    "metrics": {
        "duration_seconds": "number",
        "test_count": "number",
        "failed_tests": "number",
        "coverage": "string"
    },
    "contributors": [
        {"name": "string", "role": "string"}
    ],
    "releases": ["string"]
}

# Repair and sanitize the input
parsed_dict, success = cleaner.clean_json(malformed_input, target_schema)

print(f"Success: {success}")
import json
print(json.dumps(parsed_dict, indent=2))
```

### Response Output (Successfully Repaired):
```json
{
  "project_name": "Antigravity Pipeline",
  "build_status": "success",
  "metrics": {
    "duration_seconds": 124.5,
    "test_count": 48,
    "failed_tests": 0,
    "coverage": "98.5%"
  },
  "contributors": [
    {
      "name": "Alice",
      "role": "lead"
    },
    {
      "name": "Bob",
      "role": "reviewer"
    }
  ],
  "releases": [
    "v1.0",
    "v1.1"
  ]
}
```

---

## Configuration (`config.yaml`)

Specify settings inside the project directory:
```yaml
models:
  json_cleaner:
    path: "../../models/qwen2.5-1.5b-onnx"
    repo_id: "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX"
```
