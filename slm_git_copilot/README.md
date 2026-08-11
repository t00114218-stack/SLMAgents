# SLM Git Co-pilot

A lightweight, local CPU-optimized Conventional Commit assistant powered by a local Small Language Model (SLM) running via ONNX Runtime GenAI. It analyzes Git diff outputs and structures highly readable, descriptive, and standards-compliant conventional commit messages in real-time.

---

## Features

- **Conventional Commit Templates**: Automatically outputs messages in the standard format:
  `<type>(<scope>): <short description>`
  `[optional body details]`
- **Diff Truncation & Optimization**: Auto-truncates extremely long diff sequences (caps input context at ~4000 characters) to optimize small context windows and maintain fast generation speed.
- **Local & Offline**: Zero network latency, zero costs, and absolute security for private source code.
- **Claude-style Streaming**: Supports streaming outputs directly to terminal outputs.

---

## Installation

In your local project environment:
```bash
pip install -e ./slm_git_copilot
```

Ensure `onnxruntime-genai` is installed. It shares the central monorepo model path cached locally at `models/qwen2.5-1.5b-onnx`.

---

## API Reference

### `SLMGitCopilot`

```python
from slm_git_copilot.git_copilot import SLMGitCopilot

copilot = SLMGitCopilot(
    model_path=None,   # Path to the ONNX model directory (defaults to models/qwen2.5-1.5b-onnx)
    cache_dir=None,    # Alternative HF cache dir
    n_ctx=2048,        # Context length (defaults to 2048)
    n_threads=4        # Number of CPU threads to use for execution
)
```

#### Methods

#### `generate_commit_message(diff_text: str, stream: bool = False)`
Generates a commit message from a raw git diff content string.
- **Arguments**:
  - `diff_text` (str): Raw output from a `git diff` command.
  - `stream` (bool): If `True`, returns a token generator for real-time streaming.
- **Returns**:
  - `str` (when `stream=False`): The final parsed conventional commit message string.
  - `Generator` (when `stream=True`): Token yield generator.

---

## Usage Examples

### 1. Basic Commit Generation
```python
from slm_git_copilot.git_copilot import SLMGitCopilot

copilot = SLMGitCopilot()

sample_diff = """
diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,4 @@
-def add(a, b): return a + b
+def add(a, b):
+    # Add numbers
+    return a + b
"""

# Generate message
commit_msg = copilot.generate_commit_message(sample_diff)
print(commit_msg)
# Output:
# feat(src/main.py): Add a function to add two numbers
```

### 2. Auto-Truncation on Long Diffs
If you pass a massive diff, the agent prevents RAM spikes and token limits by slicing the diff intelligently:
```python
long_diff = "diff --git a/test.py b/test.py\n" + "hello\n" * 1000
commit_msg = copilot.generate_commit_message(long_diff)
print(commit_msg) # Generates successfully using context truncation
```

### 3. Real-Time Token Streaming
```python
stream = copilot.generate_commit_message(sample_diff, stream=True)
for token in stream:
    print(token, end="", flush=True)
```

---

## Configuration (`config.yaml`)

Specify settings inside the project directory:
```yaml
models:
  git_copilot:
    path: "../../models/qwen2.5-1.5b-onnx"
    repo_id: "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX"
```
