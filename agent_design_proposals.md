# Permissive Architectural Proposals: Web & Document Agent Suite
*(Strictly filtering for MIT & Apache 2.0 Licensed Models only)*

This document outlines the architectural blueprints, model choices, and agentic validation loops for five CPU-optimized local agents:
1. **SLM Document Parser**
2. **SLM Vision Parser**
3. **SLM Web Agent**
4. **SLM Web Scraper**
5. **SLM Search Orchestrator**

---

## 🏛️ Part 1: SLM Document Parser

The Document Parser processes file hierarchies (DOCX, Markdown, PDF text structures) and translates them into clean, schema-compliant JSON representations.

### 1. Model Selection (MIT/Apache 2.0 Only)
- **Microsoft Phi-3.5-mini-instruct (MIT):** 128k context window allows you to load large documents directly. Exceptionally strong reasoning and structured output syntax.
- **Google Gemma-1.1-2B-it (Apache 2.0):** 8k context window. Highly lightweight and efficient for smaller documents.

### 2. Agentic Validation Loop
```
       +--------------------------------------------+
       |             Raw Document File              |
       +---------------------+----------------------+
                             |
                             v
       +---------------------+----------------------+
       |        Markdown/Structure Extractor        |
       +---------------------+----------------------+
                             |
                             v
       +---------------------+----------------------+
       |      Phi-3.5-mini-instruct (MIT)           | <---------+
       |  - Parses structured document text         |           |
       |  - Wraps output in ```json ... ```         |           | (If invalid schema
       +---------------------+----------------------+           |  or syntax error)
                             |                                  |
                             v                                  |
       +---------------------+----------------------+           |
       |        JSON Schema & Type Validator        |-----------+
       |  - Verifies syntax and Pydantic compliance |
       +---------------------+----------------------+
                             | (Passes)
                             v
       +---------------------+----------------------+
       |             Verified Output JSON           |
       +--------------------------------------------+
```

---

## 👁️ Part 2: SLM Vision Parser

The Vision Parser extracts data and structures from visual assets (flowcharts, scanned infographic charts, diagrams, and image-based PDF tables).

### 1. Model Selection (MIT/Apache 2.0 Only)
- **Microsoft Florence-2-large (MIT):** 770M parameters. World-class OCR accuracy and layout segmentation with sub-second CPU latency.
- **Moondream2 (Apache 2.0):** 1.8B parameters. Excellent for conversational Visual Question Answering (VQA) on images.

### 2. Chained Agentic Loop
We chain **Florence-2** (MIT) for high-speed OCR/Table extraction with **Phi-3.5-mini** (MIT) for document reasoning and structure construction:
```
   +--------------------+
   |    Input Image     |
   +---------+----------+
             |
             v
   +---------+----------+
   |   Florence-2-large |
   |   (770M VLM - MIT) |
   |   - Run OCR & Table  |
   +---------+----------+
             |
             +----------------------------+
             | Raw OCR Text               | Segmented Table Markup
             v                            v
   +---------+----------------------------+---------+
   |        Phi-3.5-mini-instruct (3.8B - MIT)      |
   |  - Synthesizes OCR sections & table fields     |
   |  - Converts raw layout to beautiful JSON/MD    |
   +----------------------+-------------------------+
                          |
                          v
   +----------------------+-------------------------+
   |               Closed-Loop Validator            |
   |  - Runs formatting and calculation checks       |
   +------------------------------------------------+
```

---

## 🌐 Part 3: SLM Web Agent

The Web Agent performs local browser navigation and automation. It translates goals into navigation actions (click, type, scroll) and executes them locally.

### 1. Model Selection (MIT/Apache 2.0 Only)
- **Microsoft Phi-3.5-mini-instruct (MIT):** Serves as the main reasoning controller (ReAct loop). Analyzes page structures and plans step-by-step navigation actions.
- **Microsoft Florence-2-large (MIT):** Visual fallback. Coordinates bounding boxes of interactive elements when raw HTML source code is too complex or obfuscated.

### 2. ReAct Automation Loop
```
       +--------------------------------------------+
       |                 User Goal                  |
       +---------------------+----------------------+
                             |
                             v
       +---------------------+----------------------+
       |        Page State (HTML & Screen)          | <---------+
       |  - Stipped DOM tree                        |           |
       |  - Playwright browser snapshot             |           |
       +---------------------+----------------------+           |
                             |                                  |
                             v                                  |
       +---------------------+----------------------+           |
       |     Phi-3.5-mini Controller (ReAct)        |           |
       |  - Thought: Decide next step               |           |
       |  - Action: Select element locator          |           |
       +---------------------+----------------------+           |
                             |                                  |
                             v                                  |
       +---------------------+----------------------+           | (Loop until goal is
       |         Execution Subprocess               |           |  reached or failed)
       |  - Playwright invokes action (click, type) |-----------+
       +--------------------------------------------+
```

---

## 🔍 Part 4: SLM Web Scraper

The Web Scraper parses raw HTML strings and extracts targeted data structures into clean JSON files without external scraping dependencies.

### 1. Model Selection (MIT/Apache 2.0 Only)
- **Microsoft Phi-3.5-mini-instruct (MIT):** Enforces GBNF grammar constraints to output clean schema-defined structures from the layout.
- **BeautifulSoup4 / lxml (MIT / PSF License):** Used as pre-processing parsers to strip scripts, styles, and ads, reducing inputs by up to 90% to optimize CPU memory boundaries.

### 2. Scrape & Clean Pipeline
```
   [Raw HTML] -> [BeautifulSoup4 Parser] -> [Clean Markdown Representation]
                                                          |
                                                          v
                                            [Phi-3.5-mini-instruct]
                                            - Guided by Target JSON Schema
                                                          |
                                                          v
                                            [JSON Validator & Corrector]
                                            - Verifies types and syntax
                                                          |
                                                          v
                                                  [Clean JSON Output]
```

---

## 🔎 Part 5: SLM Search Orchestrator

The Search Orchestrator expands queries, queries local/offline search indexes, and reranks results to feed RAG systems.

### 1. Model & Package Selection (MIT/Apache 2.0 Only)
- **Microsoft Phi-3.5-mini-instruct (MIT):** Generates optimized search queries and filters irrelevant links.
- **all-MiniLM-L6-v2-ONNX (Apache 2.0):** Computes dense vectors locally on CPU for fast reranking.
- **DuckDuckGo-Search (MIT):** Free, headless library for searching the web without API registration keys.

### 2. Search & Rerank Workflow
1. **Query Expansion:** Phi-3.5-mini breaks down user intent into 3 distinct search terms.
2. **Retrieve Snippets:** DuckDuckGo-Search scrapes raw text blocks from the top 5 results.
3. **Local Embedding Rerank:** `all-MiniLM-L6-v2` embeds the query and snippets to calculate cosine similarities.
4. **Context Assembly:** Discards duplicates and merges the top-scoring text blocks into the RAG context.
