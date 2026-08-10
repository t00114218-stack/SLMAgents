# SLM Text-to-SQL

A lightweight, CPU-optimized Text-to-SQL translation agent powered by a local Small Language Model (SLM) running via ONNX Runtime GenAI.

This library also contains a fine-tuning script to train adapters on natural language text-to-SQL pairs using QLoRA/SFTTrainer.

---

## Installation

To install the package in editable mode locally:

```bash
pip install -e .
```

Ensure you have the required dependencies:

```bash
pip install onnxruntime-genai huggingface_hub pyyaml
```

---

## Local CPU Inference

The inference engine runs locally using ONNX Runtime GenAI. It is configured to automatically download and cache the `tonythethompson/Qwen2.5-1.5B-Instruct-ONNX` model in `~/.cache/slm_summarizer/qwen2.5-1.5b-onnx`, sharing the cache with other SLMAgents packages to save storage.

### Usage Example

```python
from slm_text_to_sql import SLMTextToSQL

# Initialize the agent
agent = SLMTextToSQL()

# Define schema and question
schema = """
CREATE TABLE Employees (
    EmployeeID INT PRIMARY KEY,
    FirstName VARCHAR(50),
    LastName VARCHAR(50),
    DepartmentID INT,
    Salary DECIMAL(10, 2)
);
CREATE TABLE Departments (
    DepartmentID INT PRIMARY KEY,
    DepartmentName VARCHAR(100)
);
"""

question = "Find the average salary of employees in the Sales department."

# Generate SQL query
sql_query = agent.generate_sql(schema=schema, question=question)
print("Generated SQL Query:")
print(sql_query)
```

---

## Fine-Tuning with QLoRA

The library includes a complete script to fine-tune a model (e.g. `Qwen/Qwen2.5-Coder-1.5B-Instruct`) for Text-to-SQL using QLoRA.

### Running Fine-Tuning

```bash
python -m slm_text_to_sql.fine_tune
```

### Google Colab / TPU/GPU Setup

If running in Google Colab:
1. Ensure you have the hardware accelerator set to GPU or TPU.
2. Install dependencies:
   ```bash
   pip install torch transformers datasets peft bitsandbytes trl accelerate
   ```
3. Run the training script or run the code blocks in a Jupyter notebook.

For detailed information on configuring training arguments or merging adapters, refer to the comments inside [fine_tune.py](file:///Users/revathysuryaprakash/Documents/SLMAgents/slm_text_to_sql/slm_text_to_sql/fine_tune.py).
