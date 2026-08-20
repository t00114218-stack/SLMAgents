#!/usr/bin/env python3
import os
import sys
import time
import torch
import sqlite3
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_production_schema import PRODUCTION_SCHEMA, EVALUATION_CASES, normalize_sql

SEED_DATA = """
INSERT INTO users VALUES (1, 'John Doe', 'john@example.com', 'customer', 'active');
INSERT INTO users VALUES (2, 'Jane Smith', 'jane.doe@example.com', 'customer', 'active');
INSERT INTO users VALUES (3, 'Bob Wilson', 'bob@example.com', 'admin', 'inactive');
INSERT INTO users VALUES (4, 'Alice Brown', 'alice@example.com', 'customer', 'active');

INSERT INTO user_profiles VALUES (1, 1, 'light', 'en');
INSERT INTO user_profiles VALUES (2, 2, 'dark', 'en');
INSERT INTO user_profiles VALUES (3, 3, 'light', 'es');
INSERT INTO user_profiles VALUES (4, 4, 'dark', 'fr');

INSERT INTO user_login_attempts VALUES (1, 1, 1, '2026-08-15 10:00:00');
INSERT INTO user_login_attempts VALUES (2, 2, 1, '2026-08-16 11:00:00');

INSERT INTO devices VALUES (1, 1, 'iOS', '17.0');
INSERT INTO devices VALUES (2, 2, 'Android', '14.0');

INSERT INTO categories VALUES (1, 'Electronics');
INSERT INTO categories VALUES (2, 'Apparel');

INSERT INTO products VALUES (1, 'Laptop', 1200.00, 10, 1);
INSERT INTO products VALUES (2, 'Phone', 800.00, 25, 1);
INSERT INTO products VALUES (3, 'Shirt', 45.00, 50, 2);
INSERT INTO products VALUES (4, 'Headphones', 150.00, 15, 1);

INSERT INTO orders VALUES (1, 1, 1200.00, '2026-08-01', 'delivered');
INSERT INTO orders VALUES (2, 2, 845.00, '2026-08-05', 'pending');
INSERT INTO orders VALUES (3, 1, 45.00, '2026-08-10', 'delivered');
INSERT INTO orders VALUES (5, 4, 150.00, '2026-08-12', 'pending');

INSERT INTO order_items VALUES (1, 1, 1, 1, 1200.00);
INSERT INTO order_items VALUES (2, 2, 2, 1, 800.00);
INSERT INTO order_items VALUES (3, 2, 3, 1, 45.00);
INSERT INTO order_items VALUES (4, 3, 3, 1, 45.00);

INSERT INTO payments VALUES (1, 1, 1200.00, 'credit_card', 'completed', '2026-08-01 10:05:00');
INSERT INTO payments VALUES (2, 2, 845.00, 'paypal', 'pending', '2026-08-05 11:15:00');

INSERT INTO shipments VALUES (1, 1, 'TRK12345', 'delivered', '2026-08-02 09:00:00');
INSERT INTO shipments VALUES (2, 3, 'TRK67890', 'pending', NULL);

INSERT INTO reviews VALUES (1, 1, 1, 5, 'Great laptop!', '2026-08-03');
INSERT INTO reviews VALUES (2, 2, 2, 4, 'Good phone.', '2026-08-06');

INSERT INTO cart VALUES (1, 2, '2026-08-18 12:00:00');
INSERT INTO cart_items VALUES (1, 1, 3, 2);

INSERT INTO coupons VALUES (1, 'SUMMER25', 25, 1);
INSERT INTO coupons VALUES (2, 'WELCOME10', 10, 1);

INSERT INTO support_tickets VALUES (1, 2, 'Dark mode issue', 'open', '2026-08-17 14:00:00');
"""

def compare_execution(schema: str, seed_sql: str, gold_sql: str, pred_sql: str) -> tuple[bool, bool, str]:
    if not pred_sql or not pred_sql.strip():
        return False, False, "Empty query"
    clean_pred = pred_sql.replace("```sql", "").replace("```", "").strip()
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.executescript(schema)
        cursor.executescript(seed_sql)
        conn.commit()
        
        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()
        cursor.execute(clean_pred)
        pred_res = cursor.fetchall()
        
        if "order by" in gold_sql.lower():
            is_ex = (gold_res == pred_res)
        else:
            is_ex = (sorted(map(str, gold_res)) == sorted(map(str, pred_res)))
        return True, is_ex, ""
    except sqlite3.Error as e:
        return False, False, str(e)
    except Exception as e:
        return False, False, str(e)
    finally:
        try:
            conn.close()
        except Exception:
            pass

def generate_unquantized_sql(model, tokenizer, schema, question):
    system_prompt = (
        "You are an expert SQL query writer. Follow these rules strictly:\n"
        "1. Only use tables and columns defined in the schema.\n"
        "2. Return ONLY the SQL query with no explanation or markdown."
    )
    prompt = (
        f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        f"<|im_start|>user\n### Database Schema\n{schema}\n\n### Question\n{question}\n\n### SQL Query<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.1,
            do_sample=False,
            eos_token_id=tokenizer.encode("<|im_end|>")[0]
        )
    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return generated.replace("```sql", "").replace("```", "").strip()

def run_unquantized_benchmark():
    print("=" * 65)
    print("   UNQUANTIZED ORIGINAL MODEL BENCHMARK (Qwen2.5-Coder-1.5B-Merged)")
    print("=" * 65)
    
    model_path = "models/qwen2.5_coder_text2sql_merged"
    print(f"\n[1/2] Loading unquantized PyTorch model from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()
    print("✅ Original unquantized model loaded successfully.")
    
    print(f"\n[2/2] Evaluating Execution Accuracy (EX) across {len(EVALUATION_CASES)} queries...")
    ex_matches = 0
    valid_count = 0
    total = len(EVALUATION_CASES)
    latencies = []
    
    for case in EVALUATION_CASES:
        qid = case["id"]
        question = case["question"]
        gold_sql = case["gold"]
        
        t0 = time.time()
        pred_sql = generate_unquantized_sql(model, tokenizer, PRODUCTION_SCHEMA, question)
        elapsed_ms = (time.time() - t0) * 1000
        latencies.append(elapsed_ms)
        
        is_valid, is_ex, err_msg = compare_execution(PRODUCTION_SCHEMA, SEED_DATA, gold_sql, pred_sql)
        if is_valid:
            valid_count += 1
        if is_ex:
            ex_matches += 1
            
        status = "✅ EX MATCH" if is_ex else ("⚠️ SYNTAX OK" if is_valid else "❌ FAIL")
        print(f"   Query #{qid:02d}: {status} | Latency: {elapsed_ms:.1f}ms")
        
    ex_rate = (ex_matches / total) * 100
    valid_rate = (valid_count / total) * 100
    avg_latency = sum(latencies) / len(latencies)
    
    print("\n" + "=" * 65)
    print("UNQUANTIZED MODEL BENCHMARK COMPLETE")
    print(f"   • Execution Accuracy (EX Match): {ex_matches}/{total} ({ex_rate:.2f}%)")
    print(f"   • Execution/Syntax Validity: {valid_count}/{total} ({valid_rate:.2f}%)")
    print(f"   • Avg Inference Latency: {avg_latency:.2f} ms")
    print("=" * 65)

if __name__ == "__main__":
    run_unquantized_benchmark()
