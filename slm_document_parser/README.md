# SLM Document Parser

A local CPU-optimized document structure and text parser agent powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** and **Florence-2-large** models running via ONNX Runtime GenAI. 

It handles complex document parsing workflows by combining a **Hybrid Visual OCR Pipeline** (rendering PDF/Office pages to images, detecting tables/figures, running block OCR, and assembling layouts using LLM reasoning) with native layout-aware parsing fallbacks. It also features **Semantic Graph-Chunking** to slice text into RAG-compliant chunks with cross-linked metadata, and can export results directly to Microsoft Excel.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Document Parser operates as an autonomous visual-to-text routing and validation loop:

```
+-----------------------------------------------------------+
|                      Raw Input File                       |
|           (PDF, DOCX, DOC, PPTX, PPT, TXT, MD)           |
+-----------------------------+-----------------------------+
                              |
                     [Convert to PDF / Image]
                              v
+-----------------------------------------------------------+
|             Hybrid Visual OCR Pipeline (Florence-2)       |
|  - Render PDF page index to PNG image via pypdfium2       |
|  - Run Object Detection (<OD>) to localize tables/figures  |
|  - Crop detected boxes and run local OCR / Captions       |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|              Layout Assembly & Markdown Synthesis          |
|  - Combine OCR text, markdown tables, and captions       |
|  - Self-correct formatting trace errors via Phi-3.5 ONNX  |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|               Semantic Graph Chunker & Linker             |
|  - Group text into RAG chunks (minimum 15-20 words)       |
|  - Extract headings, keywords, and product references     |
|  - Link related sibling paragraphs together               |
+-----------------------------+-----------------------------+
                              |
                              v
+-----------------------------------------------------------+
|                       Output Formats                      |
|             (Structured JSON, Excel spreadsheet)          |
+-----------------------------------------------------------+
```

### Advanced Features:
1. **Hybrid Visual OCR Pipeline**: Converts scanned pages or low-text layout pages to images using `pypdfium2`. Runs Florence-2 `<OD>` to localize tables and figures, crops and OCRs tables, captions figures, and uses the local LLM to reconstruct the page back into perfect Markdown.
2. **Office Document Conversions**: Leverages LibreOffice (`soffice --headless`) on Darwin/Linux to convert formats like `.docx`, `.doc`, `.pptx`, `.ppt` into clean PDFs for visual parsing, falling back to zip XML extractors and OLE stream readers (`olefile`).
3. **Semantic Linkage Chunking**: Divide documents by topics and paragraphs instead of simple character counts. Extract active section headings, subheadings, key terms, and map cross-linked references between sibling chunks.
4. **Excel spreadsheet export**: Save chunk tables (`[Index, Source, Heading, Subheading, Product, Related, Text]`) into `.xlsx` documents.

---

## ⚡ 2. CPU Performance Tuning Guidelines

1. **Allocating Threads (`n_threads`):**
   * Limit `n_threads` to your CPU's physical core count (excluding hyperthreads) to avoid cache thrashing and lockups.
2. **Context Window Configuration (`n_ctx`):**
   * Keep `n_ctx` as tight as possible (e.g., `4096` or `8192`) to reduce token evaluation latency.
3. **Memory Limits & Garbage Collection**:
   * Florence-2 is memory-intensive. The parser runs page extractions sequentially and cleans temporary page PNG images immediately after synthesis to keep the RAM footprint under 2.0 GB.

---

## 📂 3. API Reference

### `SLMDocumentParser`

```python
from slm_document_parser.document_parser import SLMDocumentParser

parser = SLMDocumentParser(
    model_path=None,   # Path to the ONNX model directory (defaults to models/phi-3.5-mini-instruct-onnx)
    cache_dir=None,    # Alternative HF cache dir
    n_ctx=4096,        # Context length (defaults to 4096)
    n_threads=4        # Number of CPU threads to use for execution
)
```

#### Methods

##### `extract_text(file_path: str) -> str`
Extracts layout-reconstructed markdown text from target document file. Runs the hybrid visual OCR pipeline for PDFs and converts office formats automatically if LibreOffice is present.
* **`file_path`** (*str*): Local path to target document.
* **Returns**: *str* representing document Markdown.

##### `chunk_document(file_path: str) -> list[dict]`
Extracts text and splits it into semantic chunks with metadata linkages.
* **`file_path`** (*str*): Local path to target document.
* **Returns**: *list[dict]* containing text and structured metadata headers.

##### `parse_and_chunk_stream(file_path: str) -> Generator`
Streaming generator yielding semantic chunks page-by-page as they are processed.
* **`file_path`** (*str*): Local path to target document.
* **Returns**: *Generator* yielding chunk dicts.

##### `export_chunks_to_excel(chunks: list[dict], output_path: str, append: bool = False) -> None`
Saves the extracted chunks to an Excel spreadsheet.
* **`chunks`** (*list[dict]*): Chunks generated by the parser.
* **`output_path`** (*str*): Target Excel file path.
* **`append`** (*bool*): Set to True to append to an existing Excel worksheet.

---

## 🚀 4. Usage Example

Here is an end-to-end usage example showing document text extraction, semantic chunking, and Excel sheet exporting:

```python
from slm_document_parser.document_parser import SLMDocumentParser

# Initialize the parser
parser = SLMDocumentParser()

file_path = "financial_report.pdf"

# 1. Parse document text (runs visual OCR pipeline for scanned tables)
markdown_content = parser.extract_text(file_path)
print("--- Document Markdown Output ---")
print(markdown_content[:500])

# 2. Extract semantic RAG chunks with cross-linked indexes
chunks = parser.chunk_document(file_path)

# 3. Export chunks directly to Excel
parser.export_chunks_to_excel(chunks, "rag_database.xlsx")
```

### Generated Output Chunks (JSON):
```json
[
  {
    "text": "SpaceX successfully launched the Falcon 9 rocket from Cape Canaveral Space Force Station, landing the booster return flight for the 15th time. The mission delivered communication payloads into low Earth orbit.",
    "metadata": {
      "source": "financial_report.pdf",
      "heading": "1. Launch Milestones",
      "subheading": "Falcon 9 Performance",
      "product": "SpaceX",
      "key_terms": ["Falcon 9", "Cape Canaveral", "booster"],
      "format": "pdf",
      "chunk_index": 0,
      "related_chunks": [1, 2]
    }
  }
]
```

### Generated Output Excel Spreadsheet Layout:
| Chunk Index | Source File | Heading | Subheading | Product | Related Chunks | Text |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 0 | financial_report.pdf | 1. Launch Milestones | Falcon 9 Performance | SpaceX | 1,2 | SpaceX successfully launched... |
