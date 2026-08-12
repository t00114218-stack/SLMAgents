# SLM Task Planner

A CPU-optimized local Task Planner that decomposes complex user goals into dependency DAGs and assigns them to specialized local agents.

---

## Features

- **Goal Decomposition**: Breaks down broad instructions into concrete execution steps.
- **Agent Delegation**: Identifies which local agent package is best suited to execute each step (e.g. `SLMPDFChat` or `SLMMathAgent`).
- **DAG Execution Flow**: Generates clean sequence structures with total step metrics.

---

## Installation

```bash
pip install -e ./slm_task_planner
```

---

## API Reference

### `SLMTaskPlanner`

```python
from slm_task_planner import SLMTaskPlanner

planner = SLMTaskPlanner()
```

#### `build_plan(goal: str) -> dict`
Decomposes a goal into sequential steps.
- **Arguments**:
  - `goal` (str): Target objective.
- **Returns**:
  - `dict`:
    ```python
    {
        "goal": str,                # Echo of the user's objective
        "tasks": list,              # Decomposed tasks list with steps and agents
        "total_steps": int          # Number of generated steps
    }
    ```

---

## Usage Example

```python
from slm_task_planner import SLMTaskPlanner

planner = SLMTaskPlanner()
goal_plan = planner.build_plan("Extract invoice data from PDF and perform tax calculation")

print(f"Goal: {goal_plan['goal']}")
print(f"Total steps: {goal_plan['total_steps']}")
print(f"Tasks: {goal_plan['tasks']}")
```

### Input & Output Example

#### Input (Goal):
```text
Extract PDF statistics
```

#### Output:
```json
{
  "goal": "Extract PDF statistics",
  "tasks": [
    {
      "step": 1,
      "task": "Extract layout & tabular data from document",
      "assigned_agent": "SLMPDFChat / SLMDocumentParser"
    }
  ],
  "total_steps": 1
}
```
