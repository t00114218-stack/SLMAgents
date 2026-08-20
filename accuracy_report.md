# Routing Accuracy & Stress Test Report

**Date/Time**: 2026-08-18 21:10:03

**Model**: Qwen 2.5 1.5B Instruct GGUF (Quantized Q4_K_M)

**Routing Temperature**: 0.7 (Non-deterministic)

**Total Runs**: 45 (15 cases x 3 runs each)

## Executive Summary

**Overall Accuracy**: **33.33%** (15/45 correct routing decisions)

### Accuracy by Category

| Category | Correct / Total | Accuracy % |
| :--- | :---: | :---: |
| Standard | 6/12 | 50.00% |
| Cross-Keyword | 0/12 | 0.00% |
| Stress | 6/12 | 50.00% |
| Noise/Edge | 3/9 | 33.33% |

## Detailed Test Case Results

| # | Category | Query | Expected | Runs (Actual Decisions) | Accuracy |
| :---: | :--- | :--- | :---: | :--- | :---: |
| 1 | Standard | `Write a quick Python sorting function for arrays in sort.py` | **CODING** | CODING, CODING, CODING | 100% |
| 2 | Standard | `Read hello.py file contents` | **CODING** | CODING, CODING, CODING | 100% |
| 3 | Standard | `Search codebase for database config parameters` | **RAG** | CODING, CODING, CODING | 0% |
| 4 | Standard | `What are the core differences between git and mercurial?` | **GENERAL** | CODING, CODING, CODING | 0% |
| 5 | Cross-Keyword | `explain the benefits of separating a RAG agent from a Coding agent` | **GENERAL** | CODING, CODING, CODING | 0% |
| 6 | Cross-Keyword | `Search the codebase for the function that writes hello world` | **RAG** | CODING, CODING, CODING | 0% |
| 7 | Cross-Keyword | `Write a document explaining how vector search indexing works` | **GENERAL** | CODING, CODING, CODING | 0% |
| 8 | Cross-Keyword | `Find where we implement code file reading in orchestrator.py` | **RAG** | CODING, CODING, CODING | 0% |
| 9 | Stress | `Can you check if hello.py exists and write a test file named test_hello.py if it is missing?` | **CODING** | CODING, CODING, CODING | 100% |
| 10 | Stress | `Where is llama-cpp-python imported in our code? Locate the exact file and explain it.` | **RAG** | GENERAL, GENERAL, GENERAL | 0% |
| 11 | Stress | `create a new folder, create files, write code inside them, and refactor existing functions` | **CODING** | CODING, CODING, CODING | 100% |
| 12 | Stress | `Explain how LLMs are quantized to Q4_K_M GGUF format and how LlamaGrammar constrains output schema` | **GENERAL** | CODING, CODING, CODING | 0% |
| 13 | Noise/Edge | `Hello there! How are you today? What can you do?` | **GENERAL** | CODING, CODING, CODING | 0% |
| 14 | Noise/Edge | `!!! codebase check ??? search !!! hello.py` | **RAG** | CODING, CODING, CODING | 0% |
| 15 | Noise/Edge | `write python` | **CODING** | CODING, CODING, CODING | 100% |