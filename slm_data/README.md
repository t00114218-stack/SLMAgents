# SLM Data Analyst

A local CPU-optimized Data Analyst agent designed to profile CSV files, auto-generate pandas summary scripts, and print descriptive statistics of local datasets.

---

## Features

- **Pandas Script Gen**: Auto-generates clean pandas DDL summaries.
- **Descriptive Statistics**: Analyzes column metrics, shape, and datatypes.
- **Sandboxed Operations**: Interfaces with local sandbox components to test generated pandas execution plans.

---

## Installation

```bash
pip install -e ./slm_data
```

---

## API Reference

### `SLMDataAnalyst`

```python
from slm_data import SLMDataAnalyst

analyst = SLMDataAnalyst()
```

#### `analyze_file(file_path: str, query: str) -> dict`
Runs profiling statistics on the target file.
- **Arguments**:
  - `file_path` (str): Path to local CSV file.
  - `query` (str): Task request query.
- **Returns**:
  - `dict`:
    ```python
    {
        "success": True/False,
        "file": str,                # Target file path
        "columns": list,            # Detected columns (if pandas environment ok)
        "script": str,              # Generated pandas evaluation script
        "summary": str              # Natural language summary description of metrics
    }
    ```

---

## Usage Example

```python
from slm_data import SLMDataAnalyst

analyst = SLMDataAnalyst()
csv_file = "sales.csv"

result = analyst.analyze_file(csv_file, "Summarize total sales")

print(f"Summary:\n{result['summary']}")
print(f"Python Script:\n{result['script']}")
```

### Input & Output Example

#### Input (CSV File + Query):
* **File**: `sales.csv`
* **Query**: `"summarize sales"`

#### Output:
```json
{
  "success": true,
  "file": "sales.csv",
  "columns": [],
  "script": "import pandas as pd\ndf = pd.read_csv('sales.csv')\nprint('Data summary:')\nprint(df.describe(include='all'))\n",
  "summary": "Calculated total revenue by region: East ($15,000), West ($22,000)."
}
```
