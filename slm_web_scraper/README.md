# SLM Web Scraper

A lightweight, local CPU-optimized HTML data scraper powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It uses **BeautifulSoup** to clean dirty HTML text blocks locally and compiles them into clean, structured schema-compliant JSON representations.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Web Scraper isolates product catalogs, details lists, and metadata grids from bloated raw HTML files before structuring them.

```
+----------------+
| Raw HTML Input |
+--------+-------+
         |
         v
+------------------------------------------+
| DOM Tree Loading (BeautifulSoup)         |
+--------+-------+
         |
         v
+------------------------------------------+
| Layout Sanitization & Noise Reduction    |
| (Strip script, style, header, nav, ads)  |
+--------+-------+
         |
         v
+------------------------------------------+
| Whitespace Collapse & Row Normalization  |
+--------+-------+
         |
         v
+------------------------------------------+
| Phi-3.5 Schema-Aligned Parsing           |
+--------+-------+
         |
         v
+----------+----------+
| JSON Syntax Validator   <------------------+
+----------+----------+                      |
           |                                  |
     [Is Invalid]                             |
           |                                  |
           v                                  |
+------------------------------------------+  |
| Self-Correction Feedback Generator       |--+
| (Re-query LLM with repair warnings)      |
+------------------------------------------+
```

### Sanitization Stages:
1. **Script & Style Removal:** Automatically removes `<script>`, `<style>`, `<noscript>`, `<nav>`, `<footer>`, `<header>`, and `<link>` elements. This strips out up to **80% of raw HTML bytes**, preventing input context overflow.
2. **Whitespace Collapse:** Condenses consecutive tabs and spacing characters, collapsing the DOM content into a flat, readable block.
3. **Schema Injection:** Injects user-specified output schemas into the prompt, asking the model to map variables to the schema directly from the sanitized markup.
4. **Validation Check:** Verifies JSON layout, initiating a self-correcting feedback cycle if brackets or structures fail validation.

---

## ⚡ 2. CPU Performance Tuning Guidelines

Processing raw webpage markup contains significant noise which slows down token evaluation on CPU. Use these tuning steps:

1. **Pruning Context Size:**
   * Always run `clean_html()` before passing text to the LLM. Passing raw HTML files often exceeds context limits, causing significant CPU delays or out-of-memory issues.
2. **Targeted Threading:**
   * Keep `n_threads` aligned to core count (typically `4`) to prevent threads from locking each other out.
3. **KV Cache Optimization:**
   * Uses ORT GenAI’s stateful model loader to process inputs efficiently.

---

## 🎯 3. Accuracy Optimization Tips

*   **Scraping Dynamic Tables:** Obfuscated tables or list cards are best parsed by converting cells to markdown-like tables (e.g. `Col 1 | Col 2`) during sanitization. This retains column-to-row alignments.
*   **Prompt Boundary Alignment:** Use correct Phi-3.5 chat format tags:
    ```text
    <|system|>
    You are an offline HTML scraping assistant.
    Return only a valid JSON block matching the schema.<|end|>
    <|user|>
    Sanitized HTML Content: {sanitized_html}
    Schema: {schema_dict}<|end|>
    <|assistant|>
    ```
*   **Schema Simplicity:** Keep the target schema dictionary flat. Deeply nested schemas can cause small models to struggle with structural alignment.

---

## 📂 4. API Reference

### `SLMWebScraper`

```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper(
    model_path="../../models/phi-3.5-mini-instruct-onnx",
    n_ctx=4096,
    n_threads=4
)
```

#### Methods

##### `clean_html(html_content: str) -> str`
Strips layout noise, headers, ads, scripts, and styling tags to yield a clean flat string structure.
* **`html_content`** (*str*): Raw webpage source code.
* **Returns**: *str* containing cleaned page body.

##### `scrape(html_content: str, schema_dict: dict, max_retries: int = 3) -> dict`
Strips webpage noise and compiles structured data into the target JSON layout.
* **`html_content`** (*str*): Raw webpage source code.
* **`schema_dict`** (*dict*): Targeted JSON structure blueprint.
* **`max_retries`** (*int*): Number of self-correcting schema retry cycles.
* **Returns**: *dict* of structured data.

##### `scrape_url(url: str, schema_dict: dict = None, max_retries: int = 3)`
Fetches webpage HTML, removes menus/navs/ads, and extracts data. If `schema_dict` is omitted, returns a clean raw text string.
* **`url`** (*str*): Webpage URL.
* **`schema_dict`** (*dict | None*): Targeted JSON structure schema (optional).
* **Returns**: *str* (raw clean text) or *dict* (schema-compliant parsed data).

---

## 🚀 5. Usage Examples

### Example 1: Webpage URL Scraping (Clean Text Extraction)
```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper()

# Scrapes page and removes ads, header menus, and footers automatically
clean_text = scraper.scrape_url("https://www.slmagents.ai/rag.html")
print(clean_text[:500])
```

#### Output:
```text
SLM RAG | Documentation
Home
›
SLM RAG
📚 Retrieval-Augmented Generation
SLM RAG
Answer questions from your own documents — locally, privately, with zero API costs. Supports custom tools, agentic Vector DB retrieval, and strict instruction adherence.
Get Started
GitHub Repo

Installation
Terminal
# Install from PyPI
pip install slm-rag
# Configure via env vars (optional)
export SLM_RAG_N_THREADS=8
export SLM_RAG_MAX_TOKENS=128

Configuration API
Constructor Parameters
Parameter | Type / Default | Description
model_path | str | None | Explicit path to an ONNX model directory.
cache_dir | str | None | Directory to download/cache the model.
n_ctx | int | 8192 | Context window size in tokens.
n_threads | int | 4 | CPU threads for inference.
```

### Example 2: HTML String Content Scraping (Structured JSON Extraction)
```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper()

raw_html_content = """
<div class="product-item">
  <span class="title">Local CPU Node Core-1</span>
  <span class="price">$150.00</span>
  <div class="rating">Rating: 4.8</div>
</div>
"""

target_schema = {
    "products": [
        {"product_name": "string", "price_usd": "number", "rating": "number"}
    ]
}

result = scraper.scrape(html_content=raw_html_content, schema_dict=target_schema)
print(result)
```

#### Generated Output Response:
```json
{
  "products": [
    {"product_name": "Local CPU Node Core-1", "price_usd": 150.00, "rating": 4.8}
  ]
}
```
