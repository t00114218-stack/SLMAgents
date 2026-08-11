# SLM Document Parser

A lightweight, local CPU-optimized document structure and text parser agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It extracts content from PDFs, DOCX, and text layout structures into clean, structured schema-compliant JSON representations.

---

## Features

- **MIT-Licensed & Permissive**: Exclusively uses MIT/Apache 2.0 components.
- **Multimodal Text Extraction**: Directly reads text contents from `.docx`, `.pdf`, `.md`, and `.txt` files.
- **Strict Schema Enforcement**: Guarantees that the parsed output adheres exactly to a user-provided JSON structure dictionary.
- **Local & Private**: Processes all documents offline with zero API calls.

---

## Installation

```bash
pip install -e ./slm_document_parser
```

---

## API Reference

### `SLMDocumentParser`

```python
from slm_document_parser.document_parser import SLMDocumentParser

parser = SLMDocumentParser(n_threads=4)
```

#### `parse(file_path: str, schema_dict: dict, max_retries: int = 3) -> dict`
- **Arguments**:
  - `file_path` (str): Absolute or relative path to a document file.
  - `schema_dict` (dict): Expected output attributes and key names.
- **Returns**:
  - `dict`: Parsed JSON map.
