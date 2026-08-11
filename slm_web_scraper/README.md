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
1. **Image Visual Extraction:** Automatically identifies `<img>` tags on the page, fetches the image data (resolving relative URLs and offline mockup routes), processes the image using the local `SLMVisionParser` under the `<DETAILED_CAPTION>` task, and replaces the `<img>` tag with the generated natural language description (e.g. `[Image Description: ...]`).
2. **Script & Style Removal:** Automatically removes `<script>`, `<style>`, `<noscript>`, `<nav>`, `<footer>`, `<header>`, and `<link>` elements. This strips out up to **80% of raw HTML bytes**, preventing input context overflow.
3. **Whitespace Collapse:** Condenses consecutive tabs and spacing characters, collapsing the DOM content into a flat, readable block.
4. **Schema Injection:** Injects user-specified output schemas into the prompt, asking the model to map variables to the schema directly from the sanitized markup.
5. **Validation Check:** Verifies JSON layout, initiating a self-correcting feedback cycle if brackets or structures fail validation.

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

### Example 1: Webpage Table Parsing (Natural Language Description)
Scrapes a page containing a configuration table and automatically represents it as a clean text description rather than raw pipe-delimited Markdown cells:

```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper()

# Scrapes a page containing the RAG API parameter table
clean_text = scraper.scrape_url("https://www.slmagents.ai/rag.html")
print(clean_text)
```

#### Output:
```text
SLM RAG | Documentation
Home › SLM RAG
📚 Retrieval-Augmented Generation
Answer questions from your own documents locally and privately with zero API costs.

[Table Description: The table outlines the constructor configuration parameters for the RAG API, containing four parameters:
- model_path (string or None, specifying the explicit path to an ONNX model directory)
- cache_dir (string or None, indicating the directory to cache the model)
- n_ctx (integer with a default of 8192, representing context window size in tokens)
- n_threads (integer with a default of 4, configuring the CPU thread count for inference)]
```

### Example 2: Webpage Image Parsing (OCR Visual Description)
Scrapes a webpage containing an image tag and uses the vision parser (Florence-2) to describe its contents inside the body text context:

```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper()

# Scrapes a page containing flowchart.png image tag
clean_text = scraper.scrape_url("https://www.slmagents.ai/vision_parser.html")
print(clean_text)
```

#### Output:
```text
SLM Vision Parser | Documentation
Overview: Translates image structures directly into text labels.

Below is the input flowchart diagram processed by the vision model:
[Image Description: A flowchart showing a start step ('Start Process') and a next step ('Next Step Link') connected with two arrows from the start step to the next step.]
```
