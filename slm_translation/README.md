# SLM Translation Hub

A CPU-optimized local Translation Hub. It supports dynamic language loading (NLLB-200 / IndicTrans2) based on translation source and target request pairs, prioritizing Indian languages.

---

## Features

- **Dynamic Model Loading**: Instantiates translational weights on-demand to conserve RAM.
- **Indian Languages Prioritization**: Fully optimized for Hindi, Tamil, Telugu, Marathi, and other regional scripts.
- **Offline translation**: Performs model calculations completely on local CPU.

---

## Installation

```bash
pip install -e ./slm_translation
```

---

## API Reference

### `SLMTranslationHub`

```python
from slm_translation import SLMTranslationHub

hub = SLMTranslationHub()
```

#### `translate(text: str, source_lang: str, target_lang: str) -> str`
Dynamically loads target model configurations and translates text.
- **Arguments**:
  - `text` (str): Text content to translate.
  - `source_lang` (str): Source language code (e.g. "en").
  - `target_lang` (str): Target language code (e.g. "hi").
- **Returns**:
  - `str`: Translated text string.

---

## Usage Example

```python
from slm_translation import SLMTranslationHub

hub = SLMTranslationHub()
translation = hub.translate("hello world", source_lang="en", target_lang="hi")

print(f"Translation: {translation}") # Output: नमस्ते दुनिया
```

### Input & Output Example

#### Input:
```json
{
  "text": "hello world",
  "src": "en",
  "tgt": "hi"
}
```

#### Output:
```text
"नमस्ते दुनिया"
```
