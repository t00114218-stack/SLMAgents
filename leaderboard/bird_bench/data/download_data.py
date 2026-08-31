#!/usr/bin/env python3
"""
BIRD Benchmark Dataset Preprocessor & Serializer
Loads the official BIRD Dev benchmark dataset, transforms structured schemas into standard SQL DDL,
and saves the ready-to-evaluate JSONL datasets.
"""
import os
import sys
import json
from datasets import load_dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def schema_dict_to_ddl(schema_dict: dict) -> str:
    """Converts BIRD structured schema dictionary into standard SQL DDL."""
    tables = schema_dict.get('table_names_original', [])
    cols = schema_dict.get('column_names_original', [])
    types = schema_dict.get('column_types', [])
    
    table_cols = {i: [] for i in range(len(tables))}
    for (tbl_idx, col_name), col_type in zip(cols, types):
        if tbl_idx == -1:  # '*' wildcard column
            continue
        # Clean column name and type
        clean_col = col_name.strip()
        clean_type = col_type.strip().upper() if col_type else "TEXT"
        table_cols[tbl_idx].append(f'"{clean_col}" {clean_type}')
    
    ddls = []
    for tbl_idx, tbl_name in enumerate(tables):
        col_defs = ",\n    ".join(table_cols[tbl_idx])
        ddls.append(f'CREATE TABLE "{tbl_name}" (\n    {col_defs}\n);')
    return "\n\n".join(ddls)

def download_and_prepare_bird_data():
    print("=" * 65)
    print("   BIRD-BENCH DATASET PREPARATION")
    print("=" * 65)
    
    print("\n[1/3] Fetching BIRD validation dataset from HuggingFace (1sf/bird-sql-dev-with-schema)...")
    dataset = load_dataset('1sf/bird-sql-dev-with-schema', split='validation')
    total_samples = len(dataset)
    print(f"✅ Loaded {total_samples} samples.")
    
    # Prepare samples
    prepared_samples = []
    print("\n[2/3] Transforming structured schemas to SQL DDL and normalizing entries...")
    for idx, sample in enumerate(dataset):
        schema_ddl = schema_dict_to_ddl(sample['schema']) if isinstance(sample.get('schema'), dict) else str(sample.get('schema', ''))
        
        entry = {
            "id": idx,
            "question_id": sample.get("question_id", idx),
            "db_id": sample.get("db_id", "default_db"),
            "question": sample.get("question", ""),
            "evidence": sample.get("evidence", ""),
            "gold_sql": sample.get("SQL", "").strip(),
            "difficulty": sample.get("difficulty", "moderate"),
            "schema_ddl": schema_ddl
        }
        prepared_samples.append(entry)
    
    # Save Full Dev (1,534 samples)
    full_path = os.path.join(DATA_DIR, "bird_dev_full.jsonl")
    with open(full_path, "w", encoding="utf-8") as f:
        for item in prepared_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved full BIRD Dev set ({len(prepared_samples)} samples) to: {full_path}")
    
    # Save Mini Dev (500 samples)
    mini_path = os.path.join(DATA_DIR, "bird_dev_500.jsonl")
    mini_samples = prepared_samples[:500]
    with open(mini_path, "w", encoding="utf-8") as f:
        for item in mini_samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"✅ Saved BIRD Mini-Dev set (500 samples) to: {mini_path}")
    
    print("\n[3/3] BIRD Dataset Preparation Complete!")
    return full_path, mini_path

if __name__ == "__main__":
    download_and_prepare_bird_data()
