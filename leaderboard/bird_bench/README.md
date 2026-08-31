# 🏆 SLMAgents Submission for BIRD-Bench (Text-to-SQL Leaderboard)

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Hardware: CPU Only](https://img.shields.io/badge/Hardware-100%25%20CPU%20Only-emerald.svg)](#-hardware-requirements--profiling)
[![Peak RAM: <2.0 GB](https://img.shields.io/badge/Peak%20RAM-%3C2.0%20GB-purple.svg)](#-hardware-requirements--profiling)
[![Benchmark: BIRD-Bench](https://img.shields.io/badge/Benchmark-BIRD--Bench-orange.svg)](https://bird-bench.github.io/)
[![Leaderboard Status: Ready for Submission](https://img.shields.io/badge/Submission-Ready%20for%20Leaderboard-brightgreen.svg)](#-official-leaderboard-submission-guide)

Official submission and reproducibility pipeline for **SLMAgents** evaluated on **BIRD-Bench** ([https://bird-bench.github.io/](https://bird-bench.github.io/)), the premier cross-domain benchmark for large-scale database-grounded Text-to-SQL evaluation.

---

## ⚡ System Architecture: `SLMAgents-Text2SQL-CPU`

The pipeline is engineered from the ground up for high-precision, low-latency Text-to-SQL generation on commodity CPU hardware without requiring dedicated GPU infrastructure or cloud API calls.

```
                         [ User Question + Evidence Hint ]
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
   [ Evidence Knowledge Parser ]                   [ Bidirectional Schema Linker ]
   • Terminology disambiguation                    • Foreign-key graph traversal
   • Formula & calculation rules                   • Context pruning (top-24 table retention)
                 │                                               │
                 └───────────────────────┬───────────────────────┘
                                         ▼
                     [ Dynamic Prompt Synthesis ]
                     • Abstract Few-Shot Guidance
                     • Qualified DDL & Column Type Schema
                                         │
                                         ▼
                     [ Local SLMTextToSQL on CPU ]
                     • Qwen2.5-Coder (INT4 ONNX Runtime GenAI)
                     • Greedy Deterministic Decoding (T=0.0)
                                         │
                                         ▼
                     [ Candidate SQL Query ]
                                         │
                                         ▼
                     [ Ephemeral SQLite In-Memory DB ]
                     • Schema compilation check
                     • Synthetic typed constraint verification
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
                 [ Compilation Pass ]            [ Database Error ]
                         │                               │
                         │                               ▼
                         │               [ Agentic Self-Correction ]
                         │               • Column glossaries injection
                         │               • Retry with error trace context
                         │                               │
                         └───────────────┬───────────────┘
                                         ▼
                     [ Verified SQL / predict_dev.json ]
```

---

## 🛠️ Key Engineering Optimizations

1. **Evidence-Grounded Prompting:**
   - Seamlessly extracts and injects external domain knowledge hints (`evidence`) from BIRD (e.g. calculation formulas, column aliases, status definitions) into the model's direct attention focus.
2. **Bidirectional Foreign-Key Schema Pruning:**
   - Prunes large multi-table enterprise schemas (often spanning 20–50 tables) down to candidate tables and connected foreign key bridges (default cap: 24 tables).
3. **Ephemeral SQLite In-Memory Self-Correction:**
   - Spins up an in-memory SQLite sandbox on CPU, compiles candidate SQL queries against the DDL, and validates syntax.
   - Automatically catches and self-corrects ambiguous column selections or missing join bridges prior to final submission.
4. **Quantized Local Generation on CPU:**
   - Executes via **ONNX Runtime GenAI** on multi-threaded CPU (`n_threads=4`), consuming less than **2.0 GB RAM** with sub-3-second latency per query.

---

## 🚀 Quickstart & Benchmark Reproduction

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/t00114218-stack/SLMAgents.git
cd SLMAgents

# Activate virtual environment
source .venv/bin/activate

# Install package in editable mode
pip install -e slm_text_to_sql/
```

### 2. Download BIRD Dataset Splits

The benchmark suite includes an automated downloader that pulls official BIRD validation data and formats DDL schemas:

```bash
python leaderboard/bird_bench/data/download_data.py
```

### 3. Run Benchmark Evaluation

To evaluate on the 500-question BIRD Mini-Dev set:

```bash
python leaderboard/bird_bench/run_benchmark.py --samples 500 --split mini --threads 4
```

To run a quick 20-sample validation check:

```bash
python leaderboard/bird_bench/run_benchmark.py --samples 20 --split mini
```

---

## 📦 Generated Output Files

All generated evaluation and submission artifacts are saved to `leaderboard/bird_bench/output/`:

| File | Purpose |
|---|---|
| `predict_dev.json` | **Official BIRD submission file** (mappings formatted as `SQL\t----- bird -----\tdb_id`) |
| `predict_dev_sql_only.json` | Clean SQL queries indexed by question ID |
| `predict_dev_detailed.jsonl` | Complete per-sample execution trace (Question, Evidence, Gold SQL, Pred SQL, Status, Latency) |
| `bird_metrics_summary.json` | Aggregate EX Accuracy, Syntax Validity, and Difficulty breakdown |
| `bird_submission_package.zip` | Bundled archive ready for upload to official leaderboard portal |

---

## 📬 Official Leaderboard Submission Guide

To submit results to the official **BIRD Leaderboard**:

1. **Online Submission Portal:**
   - Visit the official BIRD website: [https://bird-bench.github.io/](https://bird-bench.github.io/)
   - Navigate to the **Submission** section.
   - Upload the generated `leaderboard/bird_bench/output/predict_dev.json` and fill in the model metadata from `submission_metadata.json`.

2. **Email Submission:**
   - Send `output/bird_submission_package.zip` via email to: `bird.bench23@gmail.com`
   - Subject: `[BIRD-BENCH Submission] SLMAgents-Text2SQL-CPU`
   - Include the details from `submission_metadata.json` (model architecture, open-source repository link, CPU-only hardware profile).

---

## 💻 Hardware Requirements & Profiling

- **Inference Hardware:** 100% CPU Only (No GPU Required)
- **Supported Architectures:** Apple Silicon (M1/M2/M3/M4), x86_64 Linux / Windows
- **Peak Resident Memory (RAM):** 1,950 MB
- **Context Window:** 4,096 tokens
- **Threads:** 4 CPU threads (configurable via `--threads`)
- **License:** Apache 2.0
