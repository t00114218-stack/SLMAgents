# SLM Vision Parser

A lightweight, local CPU-optimized Visual PDF, chart, and whiteboard description agent powered by Microsoft's MIT-licensed **Florence-2-large** model. It runs offline OCR, table structures parsing, and visual region mapping locally on CPU.

---

## 🧠 1. Agentic Architecture & Workflow

The SLM Vision Parser translates visual charts, diagrams, flowcharts, and handwritten notes into structured semantic text blocks.

```
+------------------+
| Input Image File |
+--------+---------+
         |
         v
+------------------------------------------+
| Image Loading & Pixel Preprocessing      |
| (Rescale, normalize, tensor packaging)   |
+--------+---------+
         |
         v
+------------------------------------------+
| Task Prompt Formatting                   |
| (Encode task tags: <OCR>, <CAPTION> etc.) |
+--------+---------+
         |
         v
+------------------------------------------+
| Florence-2 Vision-Language Inference     |
| (PyTorch local weights CPU execution)    |
+--------+---------+
         |
         v
+------------------------------------------+
| Output Post-processing                   |
| (Bounding box coordinate scaling, etc.)   |
+--------+---------+
         |
         v
+------------------+
| Clean Text/JSON  |
+------------------+
```

### Vision Tasks Supported:
*   **`<OCR>`**: Extracts raw characters and line blocks.
*   **`<OCR_WITH_REGION>`**: Extracts visual tokens coupled with bounding boxes (`[ymin, xmin, ymax, xmax]`).
*   **`<CAPTION>`**: Generates brief 1-sentence summaries.
*   **`<DETAILED_CAPTION>`**: Extracts diagram components, connections, and relationship logs.
*   **`<MORE_DETAILED_CAPTION>`**: Deep layout summaries.

---

## ⚡ 2. CPU Performance Tuning Guidelines

Florence-2 utilizes visual transformer architectures. To run visual predictions on CPU within under **2.0 GB memory footprint**:

1. **Precision & Memory Management:**
   * Run with PyTorch float32 or bfloat16 hooks depending on CPU vector extensions (AVX-512 / AMX).
   * Free memory explicitly after heavy bulk jobs:
     ```python
     import gc, torch
     gc.collect()
     torch.cuda.empty_cache()  # If any hooks are initialized
     ```
2. **Batch Processing Strategy:**
   * Process document pages sequentially rather than in batches to avoid CPU memory spikes.
3. **Environment Setup:**
   Ensure threads do not clash:
   ```bash
   export OMP_NUM_THREADS=4
   export MKL_NUM_THREADS=4
   ```

---

## 🎯 3. Accuracy Optimization Tips

*   **Image Contrast and Scaling:** Small text in large document scans (e.g., schematics or blueprints) might get blurry. Pre-scale images to a standard width of `1024px` or `1280px` retaining aspect ratio, and apply simple grayscale contrast normalization using `Pillow` or `OpenCV` before parsing.
*   **Prompt Task Isolation:** Always wrap task tags in explicit angle brackets:
    ```python
    # Correct
    ocr_result = parser.parse_image("chart.png", task="<OCR>")
    
    # Incorrect
    ocr_result = parser.parse_image("chart.png", task="OCR")
    ```
*   **Table Resolution:** When using `<SURGICAL_TABLE>` or tabular vision queries, ensure the grid lines are visible and clean. Dark borders help visual transformers map columns correctly.

---

## 📂 4. API Reference

### `SLMVisionParser`

```python
from slm_vision_parser.vision_parser import SLMVisionParser

parser = SLMVisionParser(
    model_path="../../models/florence-2-large"
)
```

#### Methods

##### `parse_image(image_path: str, task: str = "<OCR>") -> str`
Parses target image file and executes the requested visual query.
* **`image_path`** (*str*): Local path to target image file.
* **`task`** (*str*): The specific Florence-2 task tag.
* **Returns**: *str* containing text content, coordinate points, or layout details.

---

## 🚀 5. Usage Example

Here is a realistic example using the `<DETAILED_CAPTION>` task tag to extract a detailed natural language text description from a flowchart diagram:

```python
from slm_vision_parser.vision_parser import SLMVisionParser

parser = SLMVisionParser()

# Describe the flowchart diagram in text using the local vision LLM
output = parser.parse_image("flowchart.png", task="<DETAILED_CAPTION>")

print(output)
```

### Generated Output Response:
```python
"A flowchart showing a start step ('Start Process') and a next step ('Next Step Link') connected with two arrows from the start step to the next step."
```
