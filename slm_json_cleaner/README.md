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

#### Methods

#### `clean_json(malformed_text: str, schema_dict: dict, stream: bool = False)`
Cleans broken or unstructured text to conform with `schema_dict`.
- **Arguments**:
  - `malformed_text` (str): Raw string containing malformed JSON.
  - `schema_dict` (dict): A reference dictionary representing the expected keys and value types.
  - `stream` (bool): If `True`, returns a generator yielding output tokens in real-time.
- **Returns**:
  - `tuple[dict, bool]` (when `stream=False`): `(parsed_json_dict, success_flag)`. Returns `{"raw_output": text, "error": msg}` and `False` if parsing fails completely.
  - `Generator` (when `stream=True`): Token yield generator.

---

## Usage Examples

### 1. Repairing Broken JSON Output
```python
from slm_json_cleaner.json_cleaner import SLMJSONCleaner

cleaner = SLMJSONCleaner()

broken_json = '{"name": "Agent Suite", "version": "0.1'
schema = {"name": "string", "version": "string"}

# Repair the JSON
parsed, success = cleaner.clean_json(broken_json, schema)

print(f"Success: {success}")
print(f"Parsed Dict: {parsed}")
# Output:
# Success: True
# Parsed Dict: {'name': 'Agent Suite', 'version': '0.1'}
```

### 2. Schema Compliance Alignment
You can enforce matching keys and value types:
```python
unstructured = '{"age": 30, "city": "New York'
schema = {"age": "number", "city": "string"}

parsed, success = cleaner.clean_json(unstructured, schema)
print(parsed)
# Output: {'age': 30, 'city': 'New York'}
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
