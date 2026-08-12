# SLM Math Agent

A CPU-optimized local Math Agent. It converts natural language descriptions of math problems into symbolic Python equations and executes them using `SymPy` to return exact symbolic calculations and numerical approximations.

---

## Features

- **SymPy Symbolic Solver**: Computes integrals, derivatives, limits, and algebraic solutions.
- **Natural Language Parsing**: Translates human text equations into clean pythonic math formulas.
- **Explanations Block**: Provides step-by-step resolution summaries alongside results.

---

## Installation

```bash
pip install -e ./slm_math
```

---

## API Reference

### `SLMMathAgent`

```python
from slm_math import SLMMathAgent

agent = SLMMathAgent()
```

#### `solve(problem_description: str) -> dict`
Parses mathematical queries and solves them symbolically.
- **Arguments**:
  - `problem_description` (str): Word math problem (e.g. "integrate x^2 from 0 to 3").
- **Returns**:
  - `dict`:
    ```python
    {
        "success": True/False,
        "problem": str,             # Input query
        "equation": str,            # Parsed symbolic math equation
        "result": str,              # Computed result value
        "explanation": str          # Step by step execution report
    }
    ```

---

## Usage Example

```python
from slm_math import SLMMathAgent

agent = SLMMathAgent()
result = agent.solve("integrate x^2 from 0 to 3")

print(f"Parsed Equation: {result['equation']}")
print(f"Result: {result['result']}")
print(f"Explanation:\n{result['explanation']}")
```

### Input & Output Example

#### Input (Problem):
```text
integrate x^2 from 0 to 3
```

#### Output:
```json
{
  "success": true,
  "problem": "integrate x^2 from 0 to 3",
  "equation": "integrate(x^2, 0, 3)",
  "result": "9",
  "explanation": "Step 1: Extracted symbolic equation formulation: `integrate(x^2, 0, 3)`\nStep 2: Calculated definite integral of `x**2` from 0 to 3.\nStep 3: Computed exact symbolic result: `9`.\n\nFinal Answer: 9"
}
```
