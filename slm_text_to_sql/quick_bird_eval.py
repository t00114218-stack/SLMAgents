#!/usr/bin/env python3
import os
import sys
import time
import json
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_text_to_sql import SLMTextToSQL
from run_full_benchmark import compare_execution

def quick_eval():
    print("=" * 65)
    print("   BIRD DEV FAST EVALUATION (5-SAMPLE QUICK CHECK, 8K CONTEXT)")
    print("=" * 65)
    
    print("\n[1/2] Loading ONNX Text2SQL model with n_ctx=8192...")
    agent = SLMTextToSQL(n_ctx=8192)
    print("✅ Model initialized.")
    
    print("\n[2/2] Running 5-sample evaluation on BIRD Dev queries...")
    dataset = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    dev_sample_indices = [idx for idx in range(1000, 2000) if dataset[idx]["num_joins"] >= 1][:5]
    dev_samples = dataset.select(dev_sample_indices)
    
    ex_matches = 0
    valid_count = 0
    
    for idx, sample in enumerate(dev_samples):
        question = sample["question"]
        schema = sample["schema"]
        gold_sql = sample["query"]
        
        t0 = time.time()
        try:
            pred_sql = agent.generate_sql(
                schema=schema,
                question=question,
                max_pruned_tables=32,
                max_tokens=1024,
                max_iterations=2
            )
            elapsed_ms = (time.time() - t0) * 1000
            
            is_valid, is_ex, err_msg, gold_res, pred_res = compare_execution(
                schema, "", gold_sql, pred_sql
            )
            
            if is_valid:
                valid_count += 1
            if is_ex:
                ex_matches += 1
                
            status_str = "✅ EX MATCH" if is_ex else ("⚠️ SYNTAX OK" if is_valid else f"❌ FAIL ({err_msg})")
            print(f"\n--- BIRD Sample #{idx+1} ---")
            print(f"Question: {question[:80]}...")
            print(f"PRED: {pred_sql}")
            print(f"Result: {status_str} | Latency: {elapsed_ms:.1f}ms")
        except Exception as e:
            print(f"\n--- BIRD Sample #{idx+1} ---")
            print(f"ERROR: {e}")
            
    print("\n" + "=" * 65)
    print(f"QUICK CHECK COMPLETE: EX Accuracy = {ex_matches}/5 ({ex_matches/5*100:.1f}%), Validity = {valid_count}/5 ({valid_count/5*100:.1f}%)")
    print("=" * 65)

if __name__ == "__main__":
    quick_eval()
