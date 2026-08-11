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

### 1. Generating Commit Message from Multi-File Diff
Here is an example showing conventional commit generation from a multi-file diff modifying routing logic and adding test assertions:

```python
from slm_git_copilot.git_copilot import SLMGitCopilot

copilot = SLMGitCopilot()

complex_diff = """
diff --git a/slm_core/orchestrator.py b/slm_core/orchestrator.py
index a12bc3..d45ef6 100644
--- a/slm_core/orchestrator.py
+++ b/slm_core/orchestrator.py
@@ -12,5 +12,12 @@ class SLMOrchestrator:
-        print("Running orchestrator route...")
+        logger.info("Initializing query router path routing details")
-        return self.fallback_run(query)
+        route = self.classifier.predict(query)
+        if route == "rag":
+            return self.rag_agent.query(query)
+        elif route == "sql":
+            return self.sql_agent.query(query)
+        return self.summarize_agent.query(query)

diff --git a/tests/test_orchestrator.py b/tests/test_orchestrator.py
index 987ef1..432ab1 100644
--- a/tests/test_orchestrator.py
+++ b/tests/test_orchestrator.py
@@ -2,4 +2,9 @@
-def test_orchestrator():
-    pass
+def test_orchestrator_routing():
+    orch = SLMOrchestrator()
+    assert orch.classifier is not None
+    assert orch.query("select * from logs") == "sql_result"
"""

# Generate message
commit_msg = copilot.generate_commit_message(complex_diff)
print(commit_msg)
```

#### Generated Commit Message:
```text
feat(slm_core): Implement classifier-based routing in SLMOrchestrator

- Replace generic print statement with semantic structured logger info
- Integrate query router classifier predicting 'rag' and 'sql' paths
- Add test_orchestrator_routing to verify sql query routing behavior
```

### 2. Auto-Truncation on Long Diffs
If you pass a massive diff, the agent prevents RAM spikes and token limits by slicing the diff intelligently:
```python
long_diff = "diff --git a/test.py b/test.py\n" + "hello\n" * 1000
commit_msg = copilot.generate_commit_message(long_diff)
print(commit_msg) # Generates successfully using context truncation
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

---

## 🔀 Advanced Git Workflow APIs

### 1. Auto Commit Staged Changes (`commit`)
Stages all tracked files, queries the local SLM to summarize modifications, writes a conventional commit message, and executes the commit automatically:
```python
from slm_git_copilot.git_copilot import SLMGitCopilot

copilot = SLMGitCopilot()
success, logs = copilot.commit()
print(logs)
```

### 2. Auto Merge & Conflict Resolution (`resolve_conflicts`)
Merges branches and automatically resolves code hunks containing conflict markers:
```python
# Try merging developer branch
success, status = copilot.merge("feature-branch")
if not success:
    print("Conflict encountered! Resolving...")
    results = copilot.resolve_conflicts()
    print("Resolved files:", results["resolved"])
```

---

## 🔌 VS Code Task Integration

To run Git Co-pilot workflow triggers directly inside VS Code, add the following configuration to your `.vscode/tasks.json` workspace file:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "SLM Git: Auto-Commit Changes",
      "type": "shell",
      "command": "python -c \"from slm_git_copilot.git_copilot import SLMGitCopilot; print(SLMGitCopilot().commit()[1])\"",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    },
    {
      "label": "SLM Git: Resolve Merge Conflicts",
      "type": "shell",
      "command": "python -c \"from slm_git_copilot.git_copilot import SLMGitCopilot; print(SLMGitCopilot().resolve_conflicts())\"",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

### How to Run:
1. Open the Command Palette (`Cmd+Shift+P` on Mac / `Ctrl+Shift+P` on Windows).
2. Type **"Run Task"** and select **"SLM Git: Auto-Commit Changes"** or **"SLM Git: Resolve Merge Conflicts"**.
3. The integrated terminal will display the generation progress, show the conventional commit message, and complete the action.
