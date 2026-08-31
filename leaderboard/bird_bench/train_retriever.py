#!/usr/bin/env python3
"""
Generic BM25 & Keyword Few-Shot Retriever indexed over the full 9,428 BIRD training set.
Dynamically retrieves the top-K most structurally and semantically similar golden SQL examples for any database.
"""
import os
import re
import json
import math
from collections import Counter
from typing import List, Dict, Any, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
TRAIN_DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bird_train_full.jsonl")

class TrainCorpusFewShotRetriever:
    """
    100% Generic In-Context Learning Retriever indexing 9,428 BIRD training pairs.
    """
    def __init__(self, train_data_path: str = TRAIN_DATA_PATH, max_entries: int = 9428):
        self.documents = []
        self.doc_tokens = []
        self.doc_freqs = Counter()
        self.doc_lens = []
        self.avg_doc_len = 0.0
        self.N = 0
        self._load_corpus(train_data_path, max_entries)
        
    def _tokenize(self, text: str) -> List[str]:
        return [w.lower() for w in re.findall(r'\w+', text) if len(w) >= 3]

    def _load_corpus(self, path: str, max_entries: int):
        if not os.path.exists(path):
            print(f"[TrainCorpusFewShotRetriever] Warning: Corpus file not found at {path}")
            return
            
        print(f"[TrainCorpusFewShotRetriever] Indexing training corpus from {path}...")
        count = 0
        total_len = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                q = entry.get("question", "")
                ev = entry.get("evidence", "")
                sql = entry.get("gold_sql", "")
                if not q or not sql:
                    continue
                    
                combined_text = f"{q} {ev}"
                tokens = self._tokenize(combined_text)
                
                self.documents.append(entry)
                self.doc_tokens.append(tokens)
                doc_len = len(tokens)
                self.doc_lens.append(doc_len)
                total_len += doc_len
                
                # Update document frequency
                unique_tokens = set(tokens)
                for t in unique_tokens:
                    self.doc_freqs[t] += 1
                    
                count += 1
                if count >= max_entries:
                    break
                    
        self.N = len(self.documents)
        self.avg_doc_len = total_len / max(1, self.N)
        print(f"[TrainCorpusFewShotRetriever] ✅ Successfully indexed {self.N} training pairs.")

    def retrieve(self, question: str, evidence: str = "", top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves the top_k most structurally matching training examples using BM25.
        """
        if not self.documents:
            return []
            
        query_text = f"{question} {evidence}"
        query_tokens = self._tokenize(query_text)
        if not query_tokens:
            return self.documents[:top_k]
            
        scores = [0.0] * self.N
        k1 = 1.5
        b = 0.75
        
        for q_tok in query_tokens:
            df = self.doc_freqs.get(q_tok, 0)
            if df == 0:
                continue
            # Standard BM25 IDF
            idf = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))
            
            for i in range(self.N):
                # Token frequency in document
                tf = self.doc_tokens[i].count(q_tok)
                if tf == 0:
                    continue
                doc_len = self.doc_lens[i]
                tf_norm = (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / max(1.0, self.avg_doc_len))))
                scores[i] += idf * tf_norm
                
        # Rank by BM25 score descending
        ranked_indices = sorted(range(self.N), key=lambda idx: scores[idx], reverse=True)
        top_results = []
        for idx in ranked_indices[:top_k]:
            if scores[idx] > 0.0:
                top_results.append(self.documents[idx])
                
        # If no score match, return top generic
        if not top_results:
            top_results = self.documents[:top_k]
            
        return top_results

    def format_few_shot_prompt(self, question: str, evidence: str = "", top_k: int = 3) -> str:
        """
        Formats retrieved training pairs into in-context few-shot examples for the prompt.
        """
        examples = self.retrieve(question, evidence, top_k=top_k)
        if not examples:
            return ""
            
        blocks = ["### In-Context Verified Examples from Training Corpus (Follow these patterns):"]
        for idx, ex in enumerate(examples, 1):
            ex_q = ex.get("question", "").strip()
            ex_ev = ex.get("evidence", "").strip()
            ex_sql = ex.get("gold_sql", "").replace("\n", " ").strip()
            
            block = f"Example #{idx}:\nQuestion: {ex_q}"
            if ex_ev:
                block += f"\nEvidence Hint: {ex_ev}"
            block += f"\nSQL:\n{ex_sql}"
            blocks.append(block)
            
        return "\n\n".join(blocks)


if __name__ == "__main__":
    retriever = TrainCorpusFewShotRetriever()
    test_q = "What is the highest eligible free rate for K-12 students in Alameda County?"
    test_ev = "Eligible free rate = Free Meal Count / Enrollment"
    print("\n" + "=" * 60)
    print("Testing Few-Shot Retrieval:")
    print(retriever.format_few_shot_prompt(test_q, test_ev, top_k=2))
