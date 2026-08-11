# SLM Web Agent

A lightweight, local CPU-optimized web browser automation companion agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It crawls pages, extracts structural links and interactive buttons, and executes browser navigation steps locally using **Playwright**.

---

## Features

- **MIT-Licensed & Permissive**: Exclusively uses MIT/Apache 2.0 components.
- **Playwright Browser Integration**: Integrates directly with Chromium web browsers.
- **ReAct Automation Loop**: Thinks inside `<thought>` tags to plan multi-step browsing behaviors.
- **Local & Secure**: Browser states and inputs are handled fully locally on device.

---

## Installation

```bash
pip install -e ./slm_web_agent
```

---

## API Reference

### `SLMWebAgent`

```python
from slm_web_agent.web_agent import SLMWebAgent

agent = SLMWebAgent()
```

#### `browse(goal: str, start_url: str, max_steps: int = 3) -> dict`
- **Arguments**:
  - `goal` (str): Task goal (e.g. "find contact email").
  - `start_url` (str): URL to initialize navigation.
  - `max_steps` (int): Number of page transition steps allowed.
- **Returns**:
  - `dict`: Contains execution history, final destination URL, and text output summary.
