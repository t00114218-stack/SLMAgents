# SLM CLI Agent

A lightweight, local CPU-optimized command line interface companion agent powered by a local Small Language Model (SLM) running via ONNX Runtime GenAI. It translates natural language instructions into precise system shell commands, explains what they do, and executes them safely through an isolation layer with built-in command security constraints.

---

## Features

- **Local CPU Execution**: Optimized to run with less than 2.0 GB RAM using INT4-quantized Qwen2.5-1.5B-Instruct-ONNX.
- **Natural Language to CLI**: Converts developer instructions (e.g. "kill process on port 8000") into correct syntax-compliant shell commands.
- **Command Security Sandbox**: Detects and automatically blocks dangerous destructive actions (e.g., `rm -rf /`, formatting drives, reboots) before running them.
- **Explainable CLI**: Explains the proposed command and its options prior to execution.
- **Streaming & Thought Process**: Supports streaming responses, outputting the model's step-by-step reasoning inside `<thought>` tags before the final output.

---

## Installation

In your local project environment:
```bash
pip install -e ./slm_cli_agent
```

Ensure `onnxruntime-genai` is installed. It uses the shared monorepo model path cached locally at `models/qwen2.5-1.5b-onnx`.

---

## API Reference

### `SLMCLIAgent`

```python
from slm_cli_agent.cli_agent import SLMCLIAgent

agent = SLMCLIAgent(
    model_path=None,   # Path to the ONNX model directory (defaults to models/qwen2.5-1.5b-onnx)
    cache_dir=None,    # Alternative HF cache dir
    n_ctx=2048,        # Context length (defaults to 2048)
    n_threads=4        # Number of CPU threads to use for execution
)
```

#### Methods

#### `generate_command(query: str, stream: bool = False)`
Translates natural language to command line input.
- **Arguments**:
  - `query` (str): Natural language instruction.
  - `stream` (bool): If `True`, returns a generator that yields decoded output tokens in real-time. If `False`, runs to completion.
- **Returns**:
  - `tuple[str, str]`: `(extracted_command, full_response)` if `stream=False`.
  - `Generator`: A generator of tokens if `stream=True`.

#### `execute_command(cmd: str) -> tuple[int, str, str]`
Executes the proposed command.
- **Arguments**:
  - `cmd` (str): The raw command line script to execute.
- **Returns**:
  - `tuple[int, str, str]`: `(return_code, stdout, stderr)`. Returns `-1` with a security warning in `stderr` if the command contains dangerous destructive patterns.

#### `run(query: str) -> dict`
Translates, explains, and runs the instruction in one method call.
- **Arguments**:
  - `query` (str): Natural language instruction.
- **Returns**:
  - `dict`:
    ```python
    {
        "success": True/False,
        "command": str,       # Extracted executable bash command
        "explanation": str,   # Model explanation and thought process
        "stdout": str,        # Execution standard output logs
        "stderr": str,        # Execution error logs
        "returncode": int     # Subprocess exit status code
    }
    ```

---

## Usage Examples

### 1. Unified Natural Language Execution (`run`)
```python
from slm_cli_agent.cli_agent import SLMCLIAgent

agent = SLMCLIAgent()

# Translate and execute in a single call
result = agent.run("list directory contents in clean format")

print(f"Success: {result['success']}")
print(f"Command Executed: {result['command']}")
print(f"Stdout:\n{result['stdout']}")
```

### 2. Manual Generation and Execution
```python
from slm_cli_agent.cli_agent import SLMCLIAgent

agent = SLMCLIAgent()

# Translate instruction to command
cmd, explanation = agent.generate_command("find all files ending with .py in current folder")
print(f"Generated Command: {cmd}")
# Output: find . -name "*.py"

# Safely execute it
code, stdout, stderr = agent.execute_command(cmd)
print(f"Exit code: {code}")
print(f"Output:\n{stdout}")
```

### 2. Command Safety Check in Action
```python
# The agent detects and blocks malicious or destructive sequences automatically
code, stdout, stderr = agent.execute_command("rm -rf /usr/bin")
print(code)    # Output: -1
print(stderr)  # Output: Execution Blocked: Destructive or dangerous command pattern detected.
```

### 3. Real-Time Token Streaming
```python
stream = agent.generate_command("Show docker containers running in background", stream=True)
for token in stream:
    print(token, end="", flush=True)
```

---

## Configuration (`config.yaml`)

Specify custom paths or model settings inside the project directory:
```yaml
models:
  cli_agent:
    path: "../../models/qwen2.5-1.5b-onnx"
    repo_id: "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX"
```
