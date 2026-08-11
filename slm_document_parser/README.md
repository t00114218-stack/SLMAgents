# SLM Document Parser

A local CPU-optimized document structure and text parser agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It extracts content from PDFs, DOCX, and text layout structures into clean, structured schema-compliant JSON representations.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Document Parser does not simply extract text; it operates as an autonomous data routing and validation loop.

```
+---------------------+
| Raw Document File   |
+----------+----------+
           |
           v
+------------------------------------------+
| Text Pre-processing & Layout Assembly    |
| (Extract paragraphs, parse tables, etc.)  |
+----------+----------+
           |
           v
+------------------------------------------+
| Phi-3.5 Parser Prompt Generation         |
| (Format instructions & schema metadata)  |
+----------+----------+
           |
           v
+------------------------------------------+
| Local ONNX CPU Inference                 |
+----------+----------+
           |
           v
+----------+----------+
| JSON & Schema Validation  <----------------+
+----------+----------+                      |
           |                                  |
     [Is Invalid]                             |
           |                                  |
           v                                  |
+------------------------------------------+  |
| Self-Correction Feedback Generator       |--+
| (Re-query LLM with traceback warnings)   |
+------------------------------------------+
```

### Self-Correction & Repair Loop:
1. **Extraction:** The parser extracts raw text blocks. If the file is a `.docx`, it scans paragraphs and renders tables into clear columnar strings (e.g., `Header 1 | Header 2 \n Cell 1 | Cell 2`) to preserve spatial context.
2. **Schema Integration:** The agent constructs a validation template showing the model exactly how the final JSON fields must be styled.
3. **Execution:** The ONNX engine runs inference using greedy search (temperature = 0.0) to maximize structural consistency.
4. **Validation:** If the model outputs broken bracket layouts or deviates from the target schema keys, the agent triggers a feedback prompt. It sends the bad output, the traceback description, and demands a corrected JSON block.

---

## ⚡ 2. CPU Performance Tuning Guidelines

To run document processing efficiently on typical server or workstation processors:

1. **Allocating Threads (`n_threads`):**
   * Do not exceed physical CPU cores (hyperthreads cause cache thrashing). Set `n_threads` strictly to physical core count (e.g., `4` or `8`).
   * Example:
     ```python
     parser = SLMDocumentParser(n_threads=4)
     ```
2. **Context Window Configuration (`n_ctx`):**
   * Keep `n_ctx` as tight as possible for the document size. Although Phi-3.5 supports up to 128K context tokens, larger contexts increase CPU latency.
   * If parsing documents under 10 pages, configure `n_ctx=4096` or `n_ctx=8192`.
3. **Threading Environment Settings:**
   Ensure your shell environment limits competing OpenMP pools:
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

---

## 🎯 3. Accuracy Optimization Tips

*   **Spatial Table Handling:** When extracting tables from Word or PDF layouts, flat paragraphs lose row associations. Ensure tables are processed into flat, delimited string lists (`row_cell_1 | row_cell_2`) before sending them to the model context.
*   **Prompt Boundary Alignment:** Use exact Phi-3.5 tags to isolate instructions from document text:
    ```text
    <|system|>
    Extract data into JSON matching the schema.<|end|>
    <|user|>
    Document Content: {document_text}
    Schema: {schema_dict}<|end|>
    <|assistant|>
    ```
*   **Preventing Schema Hallucinations:** If a target key is not present in the document, explicitly instruct the model to populate it as `null` or `""` instead of inventing dummy data.

---

## 📂 4. API Reference

### `SLMDocumentParser`

```python
from slm_document_parser.document_parser import SLMDocumentParser

parser = SLMDocumentParser(
    model_path="../../models/phi-3.5-mini-instruct-onnx",
    n_ctx=4096,
    n_threads=4
)
```

#### Methods

##### `parse(file_path: str, schema_dict: dict, max_retries: int = 3) -> dict`
Extracts and structures data from target path matching the JSON schema specifications.
* **`file_path`** (*str*): Path to target file (`.docx`, `.pdf`, `.txt`, `.md`).
* **`schema_dict`** (*dict*): Output JSON blueprint (keys and values indicating expected types/descriptions).
* **`max_retries`** (*int*): Number of self-correction feedback iterations.
* **Returns**: *dict* representing clean, validated JSON output.
