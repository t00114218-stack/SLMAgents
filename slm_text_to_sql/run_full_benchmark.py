#!/usr/bin/env python3
import os
import sys
import time
import re
import sqlite3
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_text_to_sql import SLMTextToSQL
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

FEW_SHOT_GUIDANCE = [
    {
        "question": "How many products are in stock for each category?",
        "sql": "SELECT category_id, SUM(stock) FROM products GROUP BY category_id;"
    },
    {
        "question": "Calculate total amount spent by customer 'John Doe'.",
        "sql": "SELECT SUM(o.total_amount) FROM orders o JOIN users u ON o.customer_id = u.id WHERE u.name = 'John Doe';"
    },
    {
        "question": "List products that were ordered by users using an 'Android' device.",
        "sql": "SELECT DISTINCT p.name FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN devices d ON o.customer_id = d.user_id WHERE d.platform = 'Android';"
    },
    {
        "question": "Count the number of reviews for each product.",
        "sql": "SELECT product_id, COUNT(*) AS review_count FROM reviews GROUP BY product_id;"
    },
    {
        "question": "How many devices does each user have?",
        "sql": "SELECT user_id, COUNT(*) AS device_count FROM devices GROUP BY user_id;"
    },
    {
        "question": "Get the top 5 most expensive products.",
        "sql": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"
    }
]

def compare_execution(schema: str, seed_sql: str, gold_sql: str, pred_sql: str) -> tuple[bool, bool, str, list, list]:
    if not pred_sql or not pred_sql.strip():
        return False, False, "Empty predicted query", [], []
        
    clean_pred = pred_sql
    if "```" in clean_pred:
        clean_pred = clean_pred.replace("```sql", "").replace("```", "").strip()
        
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = OFF;")
        statements = [s.strip() for s in schema.split(";") if s.strip()]
        for stmt in statements:
            try:
                cursor.execute(stmt)
            except Exception:
                pass
        if seed_sql:
            cursor.executescript(seed_sql)
        conn.commit()
        
        # Execute GOLD query
        cursor.execute(gold_sql)
        gold_res = cursor.fetchall()
        
        # Execute PRED query
        cursor.execute(clean_pred)
        pred_res = cursor.fetchall()
        
        if "order by" in gold_sql.lower():
            is_ex_match = (gold_res == pred_res)
        else:
            is_ex_match = (sorted(map(str, gold_res)) == sorted(map(str, pred_res)))
            
        return True, is_ex_match, "", gold_res, pred_res
    except sqlite3.Error as e:
        return False, False, str(e), [], []
    except Exception as e:
        return False, False, str(e), [], []
    finally:
        try:
            conn.close()
        except Exception:
            pass

def run_benchmark():
    print("=" * 65)
    print("   SPCV TEXT2SQL BENCHMARK — OPTIMIZED (TARGET: 90%+ EX)")
    print("=" * 65)
    
    print("\n[1/3] Loading local ONNX GenAI Text2SQL model...")
    start_init = time.time()
    agent = SLMTextToSQL(n_ctx=2048)
    init_time = time.time() - start_init
    print(f"✅ Model loaded successfully in {init_time:.2f}s")
    
    print(f"\n[2/3] Evaluating Execution Accuracy (EX) on {len(EVALUATION_CASES)} queries with unpruned schema...")
    
    ex_matches = 0
    valid_count = 0
    total_cases = len(EVALUATION_CASES)
    latencies = []
    results = []
    
    for case in EVALUATION_CASES:
        qid = case["id"]
        question = case["question"]
        gold_query = case["gold"]
        
        t0 = time.time()
        try:
            # Pass max_pruned_tables=16 to preserve full 15-table schema links and few_shot_examples
            pred_query = agent.generate_sql(
                schema=PRODUCTION_SCHEMA,
                question=question,
                max_pruned_tables=16,
                few_shot_examples=FEW_SHOT_GUIDANCE,
                max_iterations=3
            )
            elapsed_ms = (time.time() - t0) * 1000
            latencies.append(elapsed_ms)
            
            is_valid, is_ex, err_msg, gold_res, pred_res = compare_execution(
                PRODUCTION_SCHEMA, SEED_DATA, gold_query, pred_query
            )
            
            if is_valid:
                valid_count += 1
            if is_ex:
                ex_matches += 1
                
            results.append({
                "id": qid,
                "question": question,
                "gold": gold_query,
                "pred": pred_query,
                "execution_match": is_ex,
                "valid": is_valid,
                "error": err_msg if not is_valid else "",
                "latency_ms": elapsed_ms
            })
            
            status_str = "✅ EX MATCH" if is_ex else ("⚠️ SYNTAX OK (EX MISMATCH)" if is_valid else "❌ FAIL")
            print(f"   Query #{qid:02d}: {status_str} | Latency: {elapsed_ms:.1f}ms")
        except Exception as e:
            print(f"   Query #{qid:02d}: ❌ ERROR ({e})")
            
    ex_rate = (ex_matches / total_cases) * 100
    valid_rate = (valid_count / total_cases) * 100
    avg_latency = sum(latencies) / len(latencies) if latencies else 0
    
    print("\n[3/3] Execution Accuracy Benchmark Summary:")
    print(f"   • Total Test Queries: {total_cases}")
    print(f"   • Execution Accuracy (EX Match): {ex_matches}/{total_cases} ({ex_rate:.2f}%)")
    print(f"   • Execution/Syntax Validity: {valid_count}/{total_cases} ({valid_rate:.2f}%)")
    print(f"   • Avg Inference Latency: {avg_latency:.2f} ms")
    
    summary = {
        "model_id": "spcv/qwen2.5_coder_text2sql_onnx",
        "architecture": "Qwen2.5-Coder-1.5B-Instruct (INT4 ONNX GenAI)",
        "total_cases": total_cases,
        "execution_accuracy_ex": f"{ex_rate:.2f}%",
        "execution_validity_rate": f"{valid_rate:.2f}%",
        "average_latency_ms": f"{avg_latency:.2f} ms",
        "detailed_results": results
    }
    
    report_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_summary.json")
    with open(report_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n📊 Benchmark summary saved to {report_file}")
    
    # Re-export leaderboard submission files
    from export_leaderboard_files import spider_file, bird_file
    return summary

if __name__ == "__main__":
    run_benchmark()
