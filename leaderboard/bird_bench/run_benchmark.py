#!/usr/bin/env python3
"""
BIRD-Bench Official Evaluation & Submission Runner
Runs inference across BIRD benchmark dataset samples, evaluates Execution Accuracy (EX),
computes execution validity, and generates official BIRD leaderboard submission files.
"""
import os
import sys
import json
import time
import argparse
import zipfile
from typing import List, Dict, Any

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

from pipeline import BIRDTextToSQLAgent
from evaluate_metrics import evaluate_prediction, compute_aggregate_metrics

def run_bird_benchmark(
    dataset_file: str,
    samples_limit: int = 50,
    output_dir: str = OUTPUT_DIR,
    n_threads: int = 4,
    max_pruned_tables: int = 24
):
    print("=" * 70)
    print("   🏆 BIRD-BENCH (BIg Bench for Large-scale Database Text-to-SQL)")
    print("   Official Evaluation & Submission Pipeline — 100% CPU Inference")
    print("=" * 70)
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(dataset_file):
        # Auto-download if missing
        from data.download_data import download_and_prepare_bird_data
        download_and_prepare_bird_data()
        
    print(f"\n[1/4] Loading BIRD dataset split from: {dataset_file}...")
    dataset_samples = []
    with open(dataset_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset_samples.append(json.loads(line))
                
    total_available = len(dataset_samples)
    eval_samples = dataset_samples[:samples_limit] if samples_limit > 0 else dataset_samples
    total_eval = len(eval_samples)
    print(f"✅ Loaded {total_available} total entries. Selected {total_eval} samples for evaluation.")
    
    print(f"\n[2/4] Initializing local ONNX SLMTextToSQL Agent (threads={n_threads})...")
    agent = BIRDTextToSQLAgent(n_ctx=4096, n_threads=n_threads)
    print("✅ Model loaded successfully.")
    
    print(f"\n[3/4] Running Neural Inference & In-Memory Execution Verification ({total_eval} samples)...")
    print("-" * 70)
    
    detailed_results = []
    bird_official_submission_dict = {}
    bird_clean_sql_dict = {}
    
    latencies = []
    ex_count = 0
    valid_count = 0
    
    t_start = time.time()
    
    for idx, sample in enumerate(eval_samples):
        qid = sample.get("question_id", idx)
        db_id = sample.get("db_id", "default_db")
        question = sample.get("question", "")
        evidence = sample.get("evidence", "")
        gold_sql = sample.get("gold_sql", "")
        difficulty = sample.get("difficulty", "moderate")
        schema_ddl = sample.get("schema_ddl", "")
        
        # Run inference through pipeline
        try:
            gen_res = agent.generate(
                schema_ddl=schema_ddl,
                question=question,
                evidence=evidence,
                max_iterations=3,
                max_pruned_tables=max_pruned_tables
            )
            pred_sql = gen_res["sql"]
            latency_ms = gen_res["latency_ms"]
        except Exception as e:
            pred_sql = ""
            latency_ms = 0.0
            print(f"⚠️ Error on sample {idx}: {e}")
            
        latencies.append(latency_ms)
        
        # Evaluate against gold SQL
        eval_res = evaluate_prediction(schema_ddl, gold_sql, pred_sql)
        is_ex = eval_res["is_ex"]
        is_valid = eval_res["is_valid"]
        is_em = eval_res["is_em"]
        
        if is_ex:
            ex_count += 1
        if is_valid:
            valid_count += 1
            
        # Official BIRD leaderboard format: "<pred_sql>\t----- ----- -----\t<db_id>" or "<pred_sql>\t----- bird -----\t<db_id>"
        bird_official_submission_dict[str(idx)] = f"{pred_sql}\t----- bird -----\t{db_id}"
        bird_clean_sql_dict[str(idx)] = pred_sql
        
        record = {
            "index": idx,
            "question_id": qid,
            "db_id": db_id,
            "difficulty": difficulty,
            "question": question,
            "evidence": evidence,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "is_ex": is_ex,
            "is_valid": is_valid,
            "is_em": is_em,
            "pred_error": eval_res.get("pred_err", ""),
            "latency_ms": round(latency_ms, 2)
        }
        detailed_results.append(record)
        
        status_icon = "✅ EX" if is_ex else ("⚠️ VALID" if is_valid else "❌ FAIL")
        print(f"[{idx+1:03d}/{total_eval:03d}] DB: {db_id:<20} | Diff: {difficulty:<11} | {status_icon} | Latency: {latency_ms:.1f}ms")
        
    total_elapsed = time.time() - t_start
    avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
    
    # Compute aggregate metrics
    metrics = compute_aggregate_metrics(detailed_results)
    metrics["performance_profiling"] = {
        "total_eval_time_seconds": round(total_elapsed, 2),
        "average_query_latency_ms": round(avg_latency, 2),
        "queries_per_second": round(total_eval / total_elapsed, 2) if total_elapsed > 0 else 0.0
    }
    
    print("\n" + "=" * 70)
    print("   📊 BIRD-BENCH EVALUATION SUMMARY")
    print("=" * 70)
    print(f"   • Total Evaluated Samples:      {total_eval}")
    print(f"   • Execution Accuracy (EX):       {metrics['execution_accuracy_ex']} ({ex_count}/{total_eval})")
    print(f"   • Valid SQL / Syntax Rate:       {metrics['valid_sql_rate']} ({valid_count}/{total_eval})")
    print(f"   • Exact Match (EM) Rate:         {metrics['exact_match_em']}")
    print(f"   • Average Query Latency:         {avg_latency:.2f} ms")
    print(f"   • Total Evaluation Runtime:      {total_elapsed:.2f} s")
    
    print("\n   📈 Difficulty Breakdown:")
    for diff, stats in metrics.get("difficulty_breakdown", {}).items():
        print(f"     - {diff.capitalize():<12}: EX = {stats['ex_accuracy']:<8} | Valid = {stats['validity_rate']:<8} (Count: {stats['count']})")
    print("=" * 70)
    
    # Save outputs
    print(f"\n[4/4] Generating official submission package in: {output_dir}...")
    
    # 1. Official BIRD submission JSON
    predict_dev_path = os.path.join(output_dir, "predict_dev.json")
    with open(predict_dev_path, "w", encoding="utf-8") as f:
        json.dump(bird_official_submission_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated official BIRD submission file: {predict_dev_path}")
    
    # 2. Clean SQL prediction JSON
    predict_sql_path = os.path.join(output_dir, "predict_dev_sql_only.json")
    with open(predict_sql_path, "w", encoding="utf-8") as f:
        json.dump(bird_clean_sql_dict, f, indent=2, ensure_ascii=False)
    print(f"✅ Generated clean SQL prediction file: {predict_sql_path}")
    
    # 3. Detailed per-query trace JSONL
    detailed_path = os.path.join(output_dir, "predict_dev_detailed.jsonl")
    with open(detailed_path, "w", encoding="utf-8") as f:
        for r in detailed_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"✅ Generated detailed trace file: {detailed_path}")
    
    # 4. Metrics Summary JSON
    summary_path = os.path.join(output_dir, "bird_metrics_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"✅ Generated metrics summary file: {summary_path}")
    
    # 5. Create ready-to-submit ZIP package
    zip_pkg_path = os.path.join(output_dir, "bird_submission_package.zip")
    with zipfile.ZipFile(zip_pkg_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(predict_dev_path, arcname="predict_dev.json")
        meta_file = os.path.join(SCRIPT_DIR, "submission_metadata.json")
        if os.path.exists(meta_file):
            zipf.write(meta_file, arcname="submission_metadata.json")
        zipf.write(summary_path, arcname="bird_metrics_summary.json")
    print(f"📦 Created official BIRD submission ZIP: {zip_pkg_path}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run BIRD benchmark evaluation and generate submission files.")
    parser.add_argument("--samples", type=int, default=50, help="Number of samples to evaluate (default: 50, use -1 for all).")
    parser.add_argument("--split", type=str, default="mini", choices=["mini", "full"], help="Dataset split: 'mini' (500 samples) or 'full' (1534 samples).")
    parser.add_argument("--threads", type=int, default=8, help="CPU inference threads (default: 8)")
    parser.add_argument("--pruned_tables", type=int, default=24, help="Max pruned tables per schema.")
    
    args = parser.parse_args()
    
    dataset_filename = "bird_dev_500.jsonl" if args.split == "mini" else "bird_dev_full.jsonl"
    dataset_file_path = os.path.join(DATA_DIR, dataset_filename)
    
    run_bird_benchmark(
        dataset_file=dataset_file_path,
        samples_limit=args.samples,
        n_threads=args.threads,
        max_pruned_tables=args.pruned_tables
    )
