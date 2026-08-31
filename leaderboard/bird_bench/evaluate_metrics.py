#!/usr/bin/env python3
"""
BIRD-Bench Evaluation Metrics Engine
Computes Execution Accuracy (EX), Valid Execution Rate, Exact Match (EM),
and per-difficulty breakdowns according to BIRD benchmark standards.
"""
import re
import math
import sqlite3
from typing import Tuple, List, Dict, Any, Optional

def normalize_sql(sql: str) -> str:
    """Normalizes SQL query string for structural comparison."""
    if not sql:
        return ""
    sql = sql.lower().strip().rstrip(";")
    # Strip backticks, quotes around identifiers
    sql = re.sub(r'[`"\[\]]', '', sql)
    # Normalize whitespaces
    sql = re.sub(r'\s+', ' ', sql)
    # Normalize operator spaces
    sql = re.sub(r'\s*([,()=><!+*/-])\s*', r'\1', sql)
    return sql.strip()

def float_equal(v1: Any, v2: Any, tol: float = 1e-3) -> bool:
    """Compares numbers with floating point tolerance."""
    try:
        f1 = float(v1)
        f2 = float(v2)
        return math.isclose(f1, f2, rel_tol=tol, abs_tol=tol)
    except (ValueError, TypeError):
        return str(v1).strip().lower() == str(v2).strip().lower()

def tuples_equal(t1: tuple, t2: tuple) -> bool:
    """Compares two database row tuples with type coercion and float tolerance."""
    if len(t1) != len(t2):
        return False
    for v1, v2 in zip(t1, t2):
        if not float_equal(v1, v2):
            return False
    return True

def compare_results(res1: List[tuple], res2: List[tuple]) -> bool:
    """Compares two SQL query result sets."""
    if len(res1) != len(res2):
        return False
    if not res1 and not res2:
        return True
    
    # Try exact ordered comparison first
    if all(tuples_equal(r1, r2) for r1, r2 in zip(res1, res2)):
        return True
        
    # Unordered multiset comparison
    unmatched2 = list(res2)
    for r1 in res1:
        matched = False
        for idx, r2 in enumerate(unmatched2):
            if tuples_equal(r1, r2):
                unmatched2.pop(idx)
                matched = True
                break
        if not matched:
            return False
    return len(unmatched2) == 0

def execute_query_in_memory(schema_ddl: str, query: str) -> Tuple[bool, Optional[List[tuple]], str]:
    """
    Executes a query against an ephemeral in-memory SQLite DB loaded with the schema DDL
    and mock typed rows to evaluate compilation and result equivalence.
    """
    if not query or not query.strip():
        return False, None, "Empty query"
        
    conn = None
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Load schema
        cursor.executescript(schema_ddl)
        
        # Populate tables with synthetic mock rows to test compilation & execution paths
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f'PRAGMA table_info("{table}");')
            columns = cursor.fetchall()
            col_names = []
            col_vals = []
            for col in columns:
                col_name = col[1]
                col_type = (col[2] or "").upper()
                col_names.append(f'"{col_name}"')
                if "INT" in col_type:
                    col_vals.append(1)
                elif any(t in col_type for t in ["REAL", "FLOAT", "DOUBLE", "DECIMAL", "NUMERIC"]):
                    col_vals.append(10.5)
                elif any(t in col_type for t in ["DATE", "TIME", "TIMESTAMP"]):
                    col_vals.append("'2026-08-01'")
                else:
                    col_vals.append("'sample_val'")
            if col_names:
                insert_sql = f'INSERT OR IGNORE INTO "{table}" ({", ".join(col_names)}) VALUES ({", ".join(map(str, col_vals))});'
                cursor.execute(insert_sql)
        conn.commit()
        
        # Execute query
        cursor.execute(query)
        res = cursor.fetchall()
        return True, res, ""
    except sqlite3.Error as e:
        return False, None, str(e)
    except Exception as e:
        return False, None, str(e)
    finally:
        if conn:
            conn.close()

def evaluate_prediction(schema_ddl: str, gold_sql: str, pred_sql: str) -> Dict[str, Any]:
    """
    Evaluates a single predicted SQL against gold SQL.
    Returns EX status, EM status, syntax validity, and errors.
    """
    norm_gold = normalize_sql(gold_sql)
    norm_pred = normalize_sql(pred_sql)
    is_em = (norm_gold == norm_pred)
    
    # Check syntax & execution of pred SQL
    is_valid, pred_res, pred_err = execute_query_in_memory(schema_ddl, pred_sql)
    gold_valid, gold_res, gold_err = execute_query_in_memory(schema_ddl, gold_sql)
    
    is_ex = False
    if is_em:
        is_ex = True
    elif is_valid and gold_valid and pred_res is not None and gold_res is not None:
        is_ex = compare_results(gold_res, pred_res)
        
    return {
        "is_em": is_em,
        "is_valid": is_valid,
        "is_ex": is_ex,
        "pred_err": pred_err,
        "gold_err": gold_err
    }

def compute_aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes overall and difficulty-stratified benchmark metrics."""
    total = len(results)
    if total == 0:
        return {}
        
    ex_count = sum(1 for r in results if r.get("is_ex"))
    valid_count = sum(1 for r in results if r.get("is_valid"))
    em_count = sum(1 for r in results if r.get("is_em"))
    
    # Breakdown by difficulty
    difficulty_stats = {}
    for r in results:
        diff = r.get("difficulty", "unknown").lower()
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"total": 0, "ex": 0, "valid": 0, "em": 0}
        difficulty_stats[diff]["total"] += 1
        if r.get("is_ex"):
            difficulty_stats[diff]["ex"] += 1
        if r.get("is_valid"):
            difficulty_stats[diff]["valid"] += 1
        if r.get("is_em"):
            difficulty_stats[diff]["em"] += 1
            
    diff_summary = {}
    for diff, stats in difficulty_stats.items():
        cnt = stats["total"]
        diff_summary[diff] = {
            "count": cnt,
            "ex_accuracy": f"{(stats['ex'] / cnt) * 100:.2f}%" if cnt else "0.00%",
            "validity_rate": f"{(stats['valid'] / cnt) * 100:.2f}%" if cnt else "0.00%",
            "em_accuracy": f"{(stats['em'] / cnt) * 100:.2f}%" if cnt else "0.00%"
        }
        
    return {
        "total_samples": total,
        "execution_accuracy_ex": f"{(ex_count / total) * 100:.2f}%",
        "valid_sql_rate": f"{(valid_count / total) * 100:.2f}%",
        "exact_match_em": f"{(em_count / total) * 100:.2f}%",
        "raw_counts": {
            "ex": ex_count,
            "valid": valid_count,
            "em": em_count,
            "total": total
        },
        "difficulty_breakdown": diff_summary
    }
