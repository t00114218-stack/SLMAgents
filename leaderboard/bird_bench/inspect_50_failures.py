#!/usr/bin/env python3
"""
Diagnostic failure tracer for BIRD-Bench samples 11-25.
"""
import os
import sys
import json

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(REPO_ROOT, "leaderboard", "bird_bench"))

from pipeline import BIRDTextToSQLAgent
from evaluate_metrics import evaluate_prediction

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "bird_dev_500.jsonl")

def main():
    agent = BIRDTextToSQLAgent()
    
    samples = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
                if len(samples) >= 30:
                    break
                    
    print(f"Loaded {len(samples)} samples. Diagnosing samples 11 to 25...\n")
    
    for idx in range(10, min(25, len(samples))):
        s = samples[idx]
        q = s["question"]
        ev = s.get("evidence", "")
        gold_sql = s["gold_sql"]
        schema = s["schema_ddl"]
        db_id = s.get("db_id", "")
        
        result = agent.generate(schema, q, ev)
        pred_sql = result["sql"]
        
        eval_res = evaluate_prediction(schema, gold_sql, pred_sql)
        is_ex = eval_res["is_ex"]
        is_valid = eval_res["is_valid"]
        err = eval_res.get("pred_err", "")
        
        status = "✅ EX PASS" if is_ex else ("⚠️ VALID" if is_valid else "❌ CRASH")
        print(f"[{idx+1:03d}] {status} | DB: {db_id}")
        print(f"  Q: {q}")
        if ev:
            print(f"  Evidence: {ev}")
        print(f"  GOLD: {gold_sql}")
        print(f"  PRED: {pred_sql}")
        if err:
            print(f"  ERR : {err}")
        print("-" * 70)

if __name__ == "__main__":
    main()
