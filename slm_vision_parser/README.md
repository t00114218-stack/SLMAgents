# SLM Vision Parser

A lightweight, local CPU-optimized Visual PDF, chart, and whiteboard description agent powered by Microsoft's MIT-licensed **Florence-2-large** model. It runs offline OCR, table structures parsing, and visual region mapping locally on CPU.

---

## Features

- **MIT-Licensed & Permissive**: Exclusively uses MIT/Apache 2.0 components.
- **Fast Visual Parsing**: Performs sub-second layout, visual bounding boxes mapping, and OCR extraction.
- **Local & Offline**: Runs fully offline on standard CPUs under 2.0 GB RAM memory footprints.

---

## Installation

```bash
pip install -e ./slm_vision_parser
```

---

## API Reference

### `SLMVisionParser`

```python
from slm_vision_parser.vision_parser import SLMVisionParser

parser = SLMVisionParser()
```

#### `parse_image(image_path: str, task: str = "<OCR>") -> str`
- **Arguments**:
  - `image_path` (str): Path to an image file (PNG, JPG, BMP).
  - `task` (str): Model instruction tag (e.g. `<OCR>`, `<CAPTION>`, `<DETAILED_CAPTION>`).
- **Returns**:
  - `str`: Decoded and structured output block.
