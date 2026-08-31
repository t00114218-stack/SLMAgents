# SLMAgents submission for EnterpriseRAG-Bench (500k Corpus)

This repository contains the official, reproducible pipeline for **SLMAgents** evaluated on **EnterpriseRAG-Bench** (511,962 synthetic documents, 500 benchmark questions across 10 enterprise categories).

---

## ⚡ System Architecture

* **System Name**: `SLMAgents-Hybrid-RRF-CPU`
* **Inverted Index**: Disk-backed mmap SQLite FTS5 (BM25) with 4-column weighting (`0.0 doc_id, 5.0 title, 10.0 text, 0.0 source_type`).
* **Neural Reranker**: BGE-Reranker-Base INT8 Quantized ONNX runtime (`BAAI/bge-reranker-base`) running on CPU.
* **Context Synthesis**: Grounded multi-passage evidence aggregation + local SLMRag neural answer generator.
* **Hardware Requirements**: Runs on standard CPU (4+ threads), peak RAM footprint <3.5 GB.

---

## 🚀 Quickstart: Reproducing Results in 2 Steps

### 1. Environment Setup

```bash
git clone https://github.com/<your-org-or-user>/SLMAgents.git
cd SLMAgents

# Create Python 3.11 Virtual Environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install Dependencies
pip install onnxruntime tokenizers numpy
```

### 2. Run Official 500-Question Benchmark

```bash
# Execute full 500-question evaluation
python leaderboard/enterprise_rag/run_benchmark.py --full

# Compute official metric breakdown
python leaderboard/enterprise_rag/evaluate_metrics.py
```

Outputs are automatically saved to:
* `leaderboard/enterprise_rag/output/answers.jsonl` (Onyx submission format)
* `leaderboard/enterprise_rag/output/results.json` (Metric breakdown)

---

## 📊 Summary Results (500 / 500 Questions)

```text
==========================================================================================
📊 EnterpriseRAG-Bench Evaluation Results (Evaluated: 500 / 500 Questions)
==========================================================================================
Category                     | Count  | Recall@1  | Recall@3  | Recall@5  | Fact Coverage
------------------------------------------------------------------------------------------
project_related              | 40     |    90.0%  |    97.5%  |    97.5%  |        99.5%
completeness                 | 20     |    80.0%  |    85.0%  |    85.0%  |        95.5%
miscellaneous                | 20     |    75.0%  |    80.0%  |    85.0%  |        95.0%
constrained                  | 30     |    73.3%  |    76.7%  |    90.0%  |        86.3%
conflicting_info             | 20     |    70.0%  |    75.0%  |    85.0%  |        92.4%
intra_document_reasoning     | 40     |    67.5%  |    80.0%  |    82.5%  |        97.5%
basic                        | 175    |    53.7%  |    69.1%  |    72.0%  |        97.0%
semantic                     | 125    |    17.6%  |    32.0%  |    35.2%  |        93.2%
high_level                   | 10     |      N/A  |      N/A  |      N/A  |        97.5%
info_not_found (Abstentions) | 20     |      N/A  |      N/A  |      N/A  |       100.0% (Abstention Acc)
------------------------------------------------------------------------------------------
TOTAL RETRIEVAL APPLICABLE   | 470    |    52.3%  |    64.5%  |    68.1%  |        91.4%
==========================================================================================
```
