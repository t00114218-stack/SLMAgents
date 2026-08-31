# SLMAgents submission for EnterpriseRAG-Bench (500k Corpus)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: CPU Only](https://img.shields.io/badge/Hardware-100%25%20CPU%20Only-emerald.svg)](#hardware-requirements--profiling)
[![Peak RAM: <3.1 GB](https://img.shields.io/badge/Peak%20RAM-%3C3.1%20GB-purple.svg)](#hardware-requirements--profiling)
[![Fact Coverage: 91.4%](https://img.shields.io/badge/Fact%20Coverage-91.44%25-brightgreen.svg)](#-official-benchmark-results-500--500-questions)
[![Abstention Accuracy: 100%](https://img.shields.io/badge/Abstention%20Accuracy-100%25-green.svg)](#-official-benchmark-results-500--500-questions)

Official submission and reproducibility pipeline for **SLMAgents** evaluated on **EnterpriseRAG-Bench** (511,962 synthetic enterprise documents across Slack, Linear, Google Docs, Confluence, Jira, Gmail; 500 benchmark questions across 10 distinct evaluation categories).

---

## ⚡ System Architecture: `SLMAgents-Hybrid-RRF-CPU`

The pipeline is designed from the ground up for high-precision, low-latency enterprise retrieval on commodity CPU hardware without requiring dedicated GPU infrastructure.

```
                                  [ User Query ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   [ BM25 Lexical Inverted Index ]              [ BGE Dense Vector Embeddings ]
   • SQLite FTS5 (Memory-Mapped)                • BAAI/bge-small-en-v1.5 INT8 ONNX
   • Weighted: Title (5.0), Text (10.0)         • <12ms CPU Inference, 384-dim
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [ Reciprocal Rank Fusion (RRF, k=60) ]
                                         │
                                         ▼
                     [ BGE-Reranker Cross-Encoder ONNX ]
                     • BAAI/bge-reranker-base (INT8 on CPU)
                                         │
                                         ▼
                     [ 350-Token Context Distillation ]
                     • Salient Evidence Extraction
                     • Hierarchical Metadata Header Injection
                                         │
                                         ▼
                     [ Local SLMRag Generator on CPU ]
                     • Qwen2.5-Coder-3B-Instruct (INT4 Block-Wise)
                     • Greedy Decoding (T=0.0) + Strict Negative Grounding
                                         │
                                         ▼
                     [ Grounded Answer / Verified Fact Sheet ]
```

### Key Engineering Optimizations:
1. **Multi-Stage Precision Retrieval:**
   - **Lexical Retrieval:** Disk-backed memory-mapped SQLite FTS5 with weighted column BM25 scoring (`0.0 doc_id, 5.0 title, 10.0 text, 0.0 source_type`).
   - **Dense Retrieval:** INT8-quantized `BAAI/bge-small-en-v1.5` embeddings running via ONNX Runtime (<12ms, <40MB RAM).
   - **Rank Fusion:** Reciprocal Rank Fusion ($k=60$) dynamically merges lexical exact-match signals with dense semantic vectors.
   - **Neural Reranking:** INT8 `BAAI/bge-reranker-base` cross-encoder running on CPU threads.
2. **Context Distillation & Negative Grounding:**
   - Prunes retrieved candidates down to a strict **350-token window** to eliminate context distraction.
   - Prepends hierarchical metadata (`# Document > Section`) to preserve document context.
   - Uses strict negative constraints to guarantee **100.0% Abstention Accuracy** on out-of-scope or unanswerable queries.
3. **Quantized Local Generation:**
   - Powered by **`Qwen/Qwen2.5-Coder-3B-Instruct`** quantized to **INT4 (Group-Size 128, Symmetric)** via ONNX Runtime GenAI.
   - Memory bandwidth reduced by 70% (1.85 GB on disk, ~3,046 MB resident RAM), delivering a 3.2x speedup on CPU.

---

## 📊 Official Benchmark Results (500 / 500 Questions)

Evaluated across the complete held-out 500-question enterprise test set against the 511,962-document corpus:

| Category | Questions | Recall@1 | Recall@3 | Recall@5 | Fact Coverage | Abstention Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **project_related** | 40 | **90.0%** | **97.5%** | **97.5%** | **99.54%** | — |
| **completeness** | 20 | **80.0%** | **85.0%** | **85.0%** | **95.46%** | — |
| **miscellaneous** | 20 | **75.0%** | **80.0%** | **85.0%** | **95.00%** | — |
| **constrained** | 30 | **73.3%** | **76.7%** | **90.0%** | **86.25%** | — |
| **conflicting_info** | 20 | **70.0%** | **75.0%** | **85.0%** | **92.42%** | — |
| **intra_document_reasoning** | 40 | **67.5%** | **80.0%** | **82.5%** | **97.50%** | — |
| **basic** | 175 | **53.7%** | **69.1%** | **72.0%** | **96.95%** | — |
| **semantic** | 125 | **17.6%** | **32.0%** | **35.2%** | **93.22%** | — |
| **high_level** | 10 | *N/A* | *N/A* | *N/A* | **97.50%** | — |
| **info_not_found** | 20 | *N/A* | *N/A* | *N/A* | *0.0%* | **100.0%** |
| **TOTAL / OVERALL** | **500** | **52.34%** | **64.47%** | **68.09%** | **91.44%** | **100.0%** |

*(Note: `high_level` and `info_not_found` categories do not have single ground-truth document IDs; they are evaluated on multi-source fact coverage and refusal precision respectively).*

---

## ⚙️ Hardware Requirements & Profiling

The entire pipeline runs 100% locally on standard CPU hardware without GPU acceleration:

| Parameter | Specification |
| :--- | :--- |
| **Compute Device** | Multi-threaded CPU (4 to 8 threads recommended) |
| **GPU Required** | **No** (0 MB VRAM) |
| **Peak Resident RAM** | **~3,046 MB** (Model weights + Inverted Index + ONNX sessions) |
| **Average Query Latency** | **1.15s – 1.40s** per question on 8-core CPU |
| **Storage Footprint** | ~1.85 GB (INT4 Model) + Disk-backed SQLite FTS5 Index |

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/t00114218-stack/SLMAgents.git
cd SLMAgents

# Create and activate Python 3.11 virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install required dependencies
pip install onnxruntime tokenizers numpy
```

### 2. Run Official Benchmark & Evaluation

```bash
# 1. Execute full 500-question evaluation
python leaderboard/enterprise_rag/run_benchmark.py --full

# 2. Compute official metric breakdown
python leaderboard/enterprise_rag/evaluate_metrics.py
```

Outputs are automatically generated and saved to:
- **`leaderboard/enterprise_rag/output/answers.jsonl`**: Complete Onyx submission file containing retrieved contexts and generated answers.
- **`leaderboard/enterprise_rag/output/results.json`**: Machine-readable metric summary.
- **`leaderboard/enterprise_rag/submission_metadata.json`**: Official metadata descriptor.

---

## 📂 Repository File Structure

```text
leaderboard/enterprise_rag/
├── README.md                     # Official documentation & benchmark summary
├── submission_metadata.json      # Submission metadata & hardware specs
├── run_benchmark.py              # 500-question benchmark runner
├── evaluate_metrics.py           # Evaluation script for Recall@K & Fact Coverage
├── hybrid_retriever.py           # Hybrid SQLite FTS5 + BGE Reranker engine
├── data/
│   ├── questions.jsonl           # 500 held-out enterprise benchmark queries
│   └── enterprise_corpus.db      # 511,962-document SQLite FTS5 inverted index
└── output/
    ├── answers.jsonl             # Formatted benchmark responses (16.4 MB)
    └── results.json              # Evaluated metrics JSON
```

---

## 📬 Contact & Citation

- **Maintainer:** SLMAgents Core Team
- **Contact Email:** [suryaprakash.c.v@gmail.com](mailto:suryaprakash.c.v@gmail.com)
- **Repository:** [https://github.com/t00114218-stack/SLMAgents](https://github.com/t00114218-stack/SLMAgents)
- **Live Demo:** [https://huggingface.co/spaces/spcv/slm-agents](https://huggingface.co/spaces/spcv/slm-agents)
