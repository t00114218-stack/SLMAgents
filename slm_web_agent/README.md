# SLM Web Agent

A lightweight, local CPU-optimized web browser automation companion agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It crawls pages, extracts structural links and interactive buttons, and executes browser navigation steps locally using **Playwright**.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Web Agent utilizes a local **Reasoning and Acting (ReAct)** loop to interact with websites dynamically.

```
                  +--------------------------+
                  |  Goal Description & URL  |
                  +------------+-------------+
                               |
                               v
                     +---------+---------+
+------------------->|  Retrieve Page    |
|                    |  Interactive DOM  |
|                    +---------+---------+
|                              |
|                              v
|                    +---------+---------+
|                    |  Format Clickable  |
|                    |  Elements List    |
|                    +---------+---------+
|                              |
|                              v
|                    +---------+---------+
|                    |  Phi-3.5 Inference|
|                    |  (decide action)  |
|                    +---------+---------+
|                              |
|                              v
|                    +---------+---------+
|                    |  Fuzzy Target     |
|                    |  Verification     |
|                    +---------+---------+
|                              |
|                        [Target Found]
|                              |
|                              v
|                    +---------+---------+
|                    | Playwright Event  |
|                    | (Click, Type etc.)|
|                    +-------------------+
|                              |
+------------------------------+
```

### Step-by-Step Loop Execution:
1. **Interactive Element Extraction:** Playwright parses the current viewport's active links, buttons, input fields, and checkboxes, capturing their text values and link attributes.
2. **Strict Selector Prompting:** The agent formats these elements into a clean list of clickable choices. This list is passed to Phi-3.5, and the model is instructed to *only* choose targets from this list, which eliminates hallucinated selector paths.
3. **Reasoning Action:** The model generates output inside `<thought>...</thought>` tags before declaring its choice as JSON:
   ```json
   {
     "action": "click",
     "target": "Confirm Order"
   }
   ```
4. **Fuzzy Target Verification:** The agent parses the action JSON. If the chosen target doesn't exactly match the visible element text, the parser executes a case-insensitive and substring fuzzy match (e.g., matching `"confirm order"` or `"confirm"` to the actual button `"Confirm Order"`).
5. **Execution:** Playwright triggers the mouse or keyboard event, awaits page load, and begins the next loop step.

---

## ⚡ 2. CPU Performance Tuning Guidelines

Operating browsers and language models simultaneously on standard CPUs can be CPU-intensive. Use these tuning steps:

1. **Headless Execution:**
   Always run Playwright in headless mode. Headed browsers consume significant GPU/CPU cycle times.
   ```python
   # Controlled internally via:
   p.chromium.launch(headless=True)
   ```
2. **ONNX Thread Control:**
   Limit threads (`n_threads=4`) to prevent context-switching delays when pages load.
3. **Browser Timeouts:**
   Set short timeouts (e.g., `timeout=5000` ms) on selectors to fail fast and trigger prompt repair loops rather than hanging indefinitely.

---

## 🎯 3. Accuracy Optimization Tips

*   **Direct Element Listing:** The most common failure of small models is generating incorrect HTML query paths. By listing active clickable choices directly (e.g. `["Browse Products", "Checkout"]`) in the prompt, you restrict the action-space and ensure near 100% selector mapping accuracy.
*   **Prompt Boundary Alignment:** Use correct Phi-3.5 chat format tags:
    ```text
    <|system|>
    You are an offline browser automation controller agent.
    IMPORTANT: You can only interact with elements present in the Clickable Elements list.<|end|>
    <|user|>
    Goal: {goal}
    Current URL: {url}
    Clickable Elements: {valid_targets}<|end|>
    <|assistant|>
    ```
*   **Fuzzy Substring Matcher:** Implementing a fallback substring match makes the agent resilient to formatting variations (e.g. quotes, brackets).

---

## 📂 4. API Reference

### `SLMWebAgent`

```python
from slm_web_agent.web_agent import SLMWebAgent

agent = SLMWebAgent(
    model_path="../../models/phi-3.5-mini-instruct-onnx",
    n_ctx=4096,
    n_threads=4
)
```

#### Methods

##### `browse(goal: str, start_url: str, max_steps: int = 10) -> dict`
Navigates the browser step-by-step to fulfill the goal.
* **`goal`** (*str*): Objective to achieve.
* **`start_url`** (*str*): Starting URL.
* **`max_steps`** (*int*): Maximum browsing transitions allowed.
* **Returns**: *dict* containing execution success, history trace lists, final URL, and text output.
