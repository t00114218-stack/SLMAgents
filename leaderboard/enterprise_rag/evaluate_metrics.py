#!/usr/bin/env python3
"""
EnterpriseRAG-Bench Metrics Evaluator for SLMAgents
===================================================
Evaluates candidate answers (`output/answers.jsonl`) against benchmark gold annotations (`data/questions.jsonl`):
  1. Document Recall@1, Recall@3, Recall@5, Exact Set Match.
  2. Answer Fact Coverage & Grounding Score (against `answer_facts`).
  3. Abstention Precision/Recall for "Info Not Found" questions.
  4. Category-by-Category Performance Matrix (all 10 categories).
  5. Exports Onyx-compatible summary to `output/results.json`.

Usage:
  python evaluate_metrics.py
  python evaluate_metrics.py --answers output/answers.jsonl --questions data/questions.jsonl
"""

import os
import sys
import re
import json
import argparse
from typing import Dict, List, Any
from collections import defaultdict

_curr_dir = os.path.dirname(os.path.abspath(__file__))


def normalize_text(text: str) -> str:
    """Normalizes text for fuzzy token matching."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower())).strip()


def calculate_fact_coverage(candidate_answer: str, gold_facts: List[str]) -> float:
    """
    Computes lexical fact coverage of gold answer facts in candidate response.
    """
    if not gold_facts:
        return 1.0
    
    cand_norm = normalize_text(candidate_answer)
    cand_words = set(cand_norm.split())
    
    matched_facts = 0
    for fact in gold_facts:
        fact_norm = normalize_text(fact)
        fact_words = [w for w in fact_norm.split() if len(w) > 2]
        if not fact_words:
            continue
        
        # Fact is considered supported if >= 60% of its substantive words appear in the candidate answer
        found = sum(1 for w in fact_words if w in cand_words or w in cand_norm)
        if (found / len(fact_words)) >= 0.60:
            matched_facts += 1

    return matched_facts / len(gold_facts)


def evaluate(questions_path: str, answers_path: str, output_results_path: str) -> Dict[str, Any]:
    if not os.path.exists(questions_path):
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not os.path.exists(answers_path):
        raise FileNotFoundError(f"Answers file not found: {answers_path}")

    # Load questions keyed by question_id
    questions_map = {}
    with open(questions_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                questions_map[item["question_id"]] = item

    # Load candidate answers
    answers_list = []
    with open(answers_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    answers_list.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    total_evaluated = len(answers_list)
    if total_evaluated == 0:
        print("[Evaluator] No answers found in file.")
        return {}

    category_stats = defaultdict(lambda: {
        "count": 0,
        "doc_recall_1": 0,
        "doc_recall_3": 0,
        "doc_recall_5": 0,
        "doc_exact_match": 0,
        "fact_coverage_sum": 0.0,
        "abstentions": 0
    })

    overall = {
        "count": 0,
        "doc_recall_1": 0,
        "doc_recall_3": 0,
        "doc_recall_5": 0,
        "doc_exact_match": 0,
        "fact_coverage_sum": 0.0,
        "abstentions_correct": 0,
        "abstentions_total": 0
    }

    for item in answers_list:
        q_id = item.get("question_id")
        q = questions_map.get(q_id)
        if not q:
            continue

        q_type = q.get("question_type", "basic")
        expected_docs = set(q.get("expected_doc_ids", []))
        retrieved_docs = item.get("document_ids", [])
        cand_answer = item.get("answer", "")
        facts = q.get("answer_facts", [])

        stats = category_stats[q_type]
        stats["count"] += 1
        overall["count"] += 1

        # Retrieval metrics (for categories with gold documents)
        if expected_docs:
            top_1 = set(retrieved_docs[:1])
            top_3 = set(retrieved_docs[:3])
            top_5 = set(retrieved_docs[:5])

            if expected_docs.intersection(top_1):
                stats["doc_recall_1"] += 1
                overall["doc_recall_1"] += 1
            if expected_docs.intersection(top_3):
                stats["doc_recall_3"] += 1
                overall["doc_recall_3"] += 1
            if expected_docs.intersection(top_5):
                stats["doc_recall_5"] += 1
                overall["doc_recall_5"] += 1
            if expected_docs.issubset(top_5):
                stats["doc_exact_match"] += 1
                overall["doc_exact_match"] += 1

        # Answer fact coverage
        fact_cov = calculate_fact_coverage(cand_answer, facts)
        stats["fact_coverage_sum"] += fact_cov
        overall["fact_coverage_sum"] += fact_cov

        # Abstention accuracy (for info_not_found)
        if q_type == "info_not_found":
            overall["abstentions_total"] += 1
            is_abstention = any(p in cand_answer.lower() for p in [
                "not available", "not found", "unable to find", "no information", "cannot find"
            ]) or (len(retrieved_docs) == 0)
            if is_abstention:
                stats["abstentions"] += 1
                overall["abstentions_correct"] += 1

    # Format Results
    summary_categories = {}
    print("\n" + "=" * 90)
    print(f"📊 EnterpriseRAG-Bench Evaluation Results (Evaluated: {overall['count']} / 500 Questions)")
    print("=" * 90)
    header = f"{'Category':<28} | {'Count':<6} | {'Recall@1':<9} | {'Recall@3':<9} | {'Recall@5':<9} | {'Fact Coverage':<13}"
    print(header)
    print("-" * 90)

    for cat, s in sorted(category_stats.items()):
        c = s["count"]
        # Handle categories that have no gold documents
        if cat in ("high_level", "info_not_found"):
            summary_categories[cat] = {
                "count": c,
                "recall_at_1": "N/A (No gold docs)",
                "recall_at_3": "N/A (No gold docs)",
                "recall_at_5": "N/A (No gold docs)",
                "fact_coverage": round((s["fact_coverage_sum"] / c * 100) if c else 0, 2)
            }
            fc = (s["fact_coverage_sum"] / c * 100) if c else 0
            print(f"{cat:<28} | {c:<6} | {'N/A':>9} | {'N/A':>9} | {'N/A':>9} | {fc:11.1f}%")
        else:
            r1 = (s["doc_recall_1"] / c * 100) if c else 0
            r3 = (s["doc_recall_3"] / c * 100) if c else 0
            r5 = (s["doc_recall_5"] / c * 100) if c else 0
            fc = (s["fact_coverage_sum"] / c * 100) if c else 0

            summary_categories[cat] = {
                "count": c,
                "recall_at_1": round(r1, 2),
                "recall_at_3": round(r3, 2),
                "recall_at_5": round(r5, 2),
                "fact_coverage": round(fc, 2)
            }
            print(f"{cat:<28} | {c:<6} | {r1:7.1f}% | {r3:7.1f}% | {r5:7.1f}% | {fc:11.1f}%")

    print("-" * 90)
    retrieval_c = sum(s["count"] for cat, s in category_stats.items() if cat not in ("high_level", "info_not_found"))
    tot_r1 = (overall["doc_recall_1"] / retrieval_c * 100) if retrieval_c else 0
    tot_r3 = (overall["doc_recall_3"] / retrieval_c * 100) if retrieval_c else 0
    tot_r5 = (overall["doc_recall_5"] / retrieval_c * 100) if retrieval_c else 0
    tot_fc = (overall["fact_coverage_sum"] / overall["count"] * 100) if overall["count"] else 0

    print(f"{'RETRIEVAL APPLICABLE (470)':<28} | {retrieval_c:<6} | {tot_r1:7.1f}% | {tot_r3:7.1f}% | {tot_r5:7.1f}% | {tot_fc:11.1f}%")
    print("=" * 90 + "\n")

    results_data = {
        "benchmark": "EnterpriseRAG-Bench",
        "system_name": "SLMAgents-Hybrid-RRF-CPU",
        "total_evaluated": overall["count"],
        "overall_metrics": {
            "recall_at_1": round(tot_r1, 2),
            "recall_at_3": round(tot_r3, 2),
            "recall_at_5": round(tot_r5, 2),
            "fact_coverage": round(tot_fc, 2),
            "abstention_accuracy": round((overall["abstentions_correct"] / max(1, overall["abstentions_total"])) * 100, 2)
        },
        "per_category_metrics": summary_categories
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_results_path)), exist_ok=True)
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2)

    print(f"📁 Summary results JSON exported to: {output_results_path}")
    return results_data


def main():
    parser = argparse.ArgumentParser(description="Evaluate EnterpriseRAG-Bench candidate answers")
    parser.add_argument("--questions", default=os.path.join(_curr_dir, "data", "questions.jsonl"), help="Path to questions.jsonl")
    parser.add_argument("--answers", default=os.path.join(_curr_dir, "output", "answers.jsonl"), help="Path to candidate answers.jsonl")
    parser.add_argument("--output", default=os.path.join(_curr_dir, "output", "results.json"), help="Path to write results.json")

    args = parser.parse_args()
    evaluate(args.questions, args.answers, args.output)


if __name__ == "__main__":
    main()
