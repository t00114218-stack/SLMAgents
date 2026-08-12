# SLM PDF Chat

A CPU-optimized local PDF layout conversation agent. It leverages the local document parser and RAG structures to answer user questions about invoice formatting, structure, and text chunks offline.

---

## Features

- **RAG QA Integration**: Interfaces with the local vector RAG setup.
- **Layout Aware Parsing**: Uses the local document parser to process raw document files.
- **Safety Warnings**: Returns contextual alerts if files are not loaded prior to lookup.

---

## Installation

```bash
pip install -e ./slm_pdf
```

---

## API Reference

### `SLMPDFChat`

```python
from slm_pdf import SLMPDFChat

agent = SLMPDFChat()
```

#### `load(pdf_path: str)`
Loads the target PDF file into the document parser.
- **Arguments**:
  - `pdf_path` (str): Local file path to the PDF.

#### `ask(question: str) -> str`
Queries the loaded PDF content.
- **Arguments**:
  - `question` (str): Question about the PDF.
- **Returns**:
  - `str`: Text response from the RAG pipeline.

---

## Usage Example

```python
from slm_pdf import SLMPDFChat

agent = SLMPDFChat()
agent.load("sample_invoice.pdf")

answer = agent.ask("What is the total balance due?")
print(f"Answer: {answer}")
```

### Input & Output Example

#### Input (Question before load):
```text
What is total revenue?
```

#### Output:
```text
No PDF document loaded. Please call `.load(pdf_path)` first.
```
