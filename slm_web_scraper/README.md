# SLM Web Scraper

A lightweight, local CPU-optimized HTML data scraper powered by Microsoft's MIT-licensed **Phi-3.5-mini-instruct** model running via ONNX Runtime GenAI. It uses **BeautifulSoup** to clean dirty HTML text blocks locally and compiles them into clean, structured schema-compliant JSON representations.

---

## Features

- **MIT-Licensed & Permissive**: Exclusively uses MIT/Apache 2.0 components.
- **HTML DOM Sanitization**: Strips scripts, styling, and navigation blocks automatically to minimize memory overhead.
- **Target Schema Enforcement**: Maps variables and elements directly into expected target JSON dictionary formats.
- **Offline Scraper**: Executes fully offline on standard CPU without cloud scraper APIs.

---

## Installation

```bash
pip install -e ./slm_web_scraper
```

---

## API Reference

### `SLMWebScraper`

```python
from slm_web_scraper.web_scraper import SLMWebScraper

scraper = SLMWebScraper()
```

#### `clean_html(html_content: str) -> str`
Strips style and script headers, returning simplified layout text elements.

#### `scrape(html_content: str, schema_dict: dict, max_retries: int = 3) -> dict`
- **Arguments**:
  - `html_content` (str): Raw HTML source content.
  - `schema_dict` (dict): Expected key-value attributes list.
- **Returns**:
  - `dict`: Parsed and formatted output dict.
