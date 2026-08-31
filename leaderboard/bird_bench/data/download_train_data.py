#!/usr/bin/env python3
"""
Download the complete official BIRD-Bench Training Dataset (9,428 samples across 80 databases).
Saves to leaderboard/bird_bench/data/bird_train_full.jsonl
"""
import os
import json
import time
from typing import Dict, Any, List
from datasets import load_dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(DATA_DIR, "bird_train_full.jsonl")

def convert_schema_to_ddl(schema_dict: Any) -> str:
    """Converts structured schema dictionary/list to standard CREATE TABLE DDLs."""
    if not isinstance(schema_dict, dict):
        return str(schema_dict)
    ddl_statements = []
    for table_name, table_info in schema_dict.items():
        if isinstance(table_info, dict):
            columns = table_info.get("columns", [])
        elif isinstance(table_info, list):
            columns = table_info
        else:
            continue
            
        col_defs = []
        for col in columns:
            if isinstance(col, dict):
                col_name = str(col.get("name", ""))
                col_type = str(col.get("type", "TEXT"))
            elif isinstance(col, (list, tuple)) and len(col) >= 1:
                col_name = str(col[0])
                col_type = str(col[1]) if len(col) > 1 else "TEXT"
            elif isinstance(col, (str, int, float)):
                col_name = str(col)
                col_type = "TEXT"
            else:
                continue
                
            if not col_name.strip():
                continue
            if any(ch in col_name for ch in [' ', '(', ')', '/', '-', '%', '$', '#']):
                col_def = f'    "{col_name}" {col_type.upper()}'
            else:
                col_def = f'    `{col_name}` {col_type.upper()}'
            col_defs.append(col_def)
        if col_defs:
            table_ddl = f'CREATE TABLE "{table_name}" (\n' + ",\n".join(col_defs) + "\n);"
            ddl_statements.append(table_ddl)
    return "\n\n".join(ddl_statements)

def download_bird_train():
    print("=" * 70)
    print("📥 DOWNLOADING BIRD-BENCH OFFICIAL TRAINING DATASET (9,428 SAMPLES)")
    print("=" * 70)
    t0 = time.time()
    
    ds = load_dataset("1sf/bird-sql-train-with-schema", split="train")
    print(f"✅ Loaded dataset! Total samples: {len(ds)}")
            
    print(f"Writing {len(ds)} training samples to: {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f_out:
        for idx, item in enumerate(ds):
            raw_schema = item.get("schema", {})
            schema_ddl = convert_schema_to_ddl(raw_schema) if isinstance(raw_schema, dict) else str(raw_schema)
            
            entry = {
                "question_id": item.get("question_id", idx),
                "db_id": item.get("db_id", ""),
                "question": item.get("question", ""),
                "evidence": item.get("evidence", ""),
                "gold_sql": item.get("SQL", item.get("sql", item.get("query", ""))),
                "difficulty": item.get("difficulty", "simple"),
                "schema_ddl": schema_ddl
            }
            f_out.write(json.dumps(entry) + "\n")
            
    elapsed = time.time() - t0
    print(f"✅ Finished! Saved {len(ds)} training samples to {OUTPUT_FILE} in {elapsed:.2f}s.")

if __name__ == "__main__":
    download_bird_train()
