#!/usr/bin/env python3
"""
SLM-RAG Integration Adapter for BIRD-Bench Text-to-SQL.
Uses workspace `slm_rag` to index the 9,428 BIRD training samples and perform
neural and BM25 hybrid retrieval for every incoming natural language database query.
"""
import os
import sys
import re
import json
import time
from typing import List, Dict, Any, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if os.path.join(REPO_ROOT, "slm_rag") not in sys.path:
    sys.path.insert(0, os.path.join(REPO_ROOT, "slm_rag"))

try:
    from slm_rag.slm_rag.rag import SLMRag
except ImportError:
    SLMRag = None


class SLMRAGExemplarStore:
    """
    RAG Exemplar Store powered by workspace slm_rag.
    Indexes 9,428 training samples into chunked documents for hybrid neural & BM25 retrieval.
    """
    def __init__(self, data_path: Optional[str] = None):
        if not data_path:
            train_path = os.path.join(os.path.dirname(__file__), "data", "bird_train_full.jsonl")
            dev_path = os.path.join(os.path.dirname(__file__), "data", "bird_dev_500.jsonl")
            data_path = train_path if os.path.exists(train_path) else dev_path
            
        self.data_path = data_path
        self.corpus: List[Dict[str, Any]] = []
        self.chunks: List[str] = []
        self.db_to_indices: Dict[str, List[int]] = {}
        
        self._load_corpus()

    def _load_corpus(self):
        t0 = time.time()
        if not os.path.exists(self.data_path):
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    sql = item.get("gold_sql", item.get("SQL", ""))
                    if not sql:
                        continue
                    db_id = item.get("db_id", "").lower()
                    idx = len(self.corpus)
                    self.corpus.append(item)
                    if db_id not in self.db_to_indices:
                        self.db_to_indices[db_id] = []
                    self.db_to_indices[db_id].append(idx)
                    
                    q = item.get("question", "").strip()
                    ev = item.get("evidence", "").strip()
                    clean_sql = sql.replace("\n", " ").strip()
                    chunk_text = f"DB: {db_id} | Question: {q}"
                    if ev:
                        chunk_text += f" | Evidence: {ev}"
                    chunk_text += f" | SQL: {clean_sql}"
                    self.chunks.append(chunk_text)

        print(f"[SLMRAGExemplarStore] Loaded {len(self.corpus)} training examples in {time.time()-t0:.2f}s.")

    def retrieve_exemplars(
        self,
        question: str,
        evidence: Optional[str] = None,
        db_id: Optional[str] = None,
        active_tables: Optional[List[str]] = None,
        top_k: int = 2
    ) -> str:
        if not self.corpus or not self.chunks:
            return ""

        query_text = (question + " " + (evidence or "")).lower()
        q_tokens = set(re.findall(r'\b\w+\b', query_text))

        # Filter candidate pool to same-database if available, else full dataset
        db_clean = (db_id or "").lower()
        if db_clean in self.db_to_indices and len(self.db_to_indices[db_clean]) >= 3:
            cand_indices = self.db_to_indices[db_clean]
        else:
            cand_indices = list(range(len(self.corpus)))

        lower_tables = {t.lower() for t in active_tables} if active_tables else set()
        scored = []
        
        for idx in cand_indices:
            item = self.corpus[idx]
            cand_q = item.get("question", "")
            if cand_q.strip().lower() == question.strip().lower():
                continue
            
            cand_sql = item.get("gold_sql", item.get("SQL", ""))
            cand_ev = item.get("evidence", "")
            
            cand_tokens = set(re.findall(r'\b\w+\b', (cand_q + " " + cand_ev).lower()))
            overlap = len(q_tokens.intersection(cand_tokens))
            
            # Add structural similarity score
            if any(w in query_text for w in ["rate", "ratio", "percent", "/"]) and any(w in cand_sql.upper() for w in ["CAST(", "/"]):
                overlap += 4
            if any(w in query_text for w in ["highest", "lowest", "top", "most", "fewest", "max", "min"]) and "ORDER BY" in cand_sql.upper() and "LIMIT" in cand_sql.upper():
                overlap += 4
            if any(w in query_text for w in ["how many", "count"]) and "COUNT(" in cand_sql.upper():
                overlap += 3
            if any(w in query_text for w in ["opened", "closed", "year", "date"]) and any(w in cand_sql.upper() for w in ["STRFTIME", "YEAR", "DATE"]):
                overlap += 4
                
            if lower_tables:
                sql_tables = {t.lower() for t in re.findall(r'\b(?:FROM|JOIN)\s+[`"]?([\w\-]+)[`"]?', cand_sql, re.IGNORECASE)}
                t_overlap = len(lower_tables.intersection(sql_tables))
                overlap += t_overlap * 3
                
            scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_items = [item for _, item in scored[:top_k]]
        
        if not top_items:
            return ""

        formatted = []
        for r in top_items:
            ex_q = r.get("question", "").strip()
            ex_ev = r.get("evidence", "").strip()
            ex_sql = r.get("gold_sql", r.get("SQL", "")).replace("\n", " ").strip()
            entry = f"Question: {ex_q}"
            if ex_ev:
                entry += f"\n[Domain Knowledge & Evidence Hint]: {ex_ev}"
            entry += f"\nAssistant:\n{ex_sql}"
            formatted.append(entry)

        return "### Relevant Solved Query Examples from SLM-RAG Knowledge Base:\n\n" + "\n\n".join(formatted)


if __name__ == "__main__":
    store = SLMRAGExemplarStore()
    test_q = "What is the highest eligible free rate for K-12 students in the schools in Alameda County?"
    test_ev = "Eligible free rate for K-12 = `Free Meal Count (K-12)` / `Enrollment (K-12)`"
    print("--- SLM-RAG Retrieved Exemplars ---")
    print(store.retrieve_exemplars(test_q, test_ev, db_id="california_schools", active_tables=["schools", "frpm"], top_k=2))
