# SLM Embeddings Server

A CPU-optimized local Vector Embeddings Server. It supports generation of dense semantic text vectors (dimensions up to 1024) utilizing local model embeddings, prioritizing Mixbread-sdk models.

---

## Features

- **High-Dimension Embeddings**: Standardized 1024-dimension vector output for downstream search.
- **Mixbread Model Integration**: Configured to interface with local Mixbread optimization layers.
- **Cosine Similarity helpers**: Computes proximity scores between prompt query vectors and chunk context vectors.

---

## Installation

```bash
pip install -e ./slm_embeddings
```

---

## API Reference

### `SLMEmbeddingsServer`

```python
from slm_embeddings import SLMEmbeddingsServer

server = SLMEmbeddingsServer()
```

#### `embed(texts: list) -> list`
Generates vector arrays for given list of text blocks.
- **Arguments**:
  - `texts` (list): Array of strings (e.g. `["sample test"]`).
- **Returns**:
  - `list`: Array of float vector values (e.g. `[[0.012, -0.045, ...]]`).

---

## Usage Example

```python
from slm_embeddings import SLMEmbeddingsServer

server = SLMEmbeddingsServer()
texts = ["Local CPU-optimized SLM agents"]

vectors = server.embed(texts)

print(f"Dimension: {len(vectors[0])}") # Output: 1024
```

### Input & Output Example

#### Input (Text):
```text
"sample test"
```

#### Output:
```text
"Vector dimension check: 1024"
```
