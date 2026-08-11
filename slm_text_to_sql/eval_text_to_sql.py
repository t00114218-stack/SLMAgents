import os
import sys
import re
import sqlite3
from datasets import load_dataset

# Ensure local path is prioritized to import local slm_text_to_sql
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_text_to_sql import SLMTextToSQL

def normalize_sql(sql: str) -> str:
    """
    Normalizes a SQL query for a fairer text comparison by:
    1. Converting to lowercase
    2. Stripping leading/trailing spaces and semicolons
    3. Collapsing multiple spaces/newlines into a single space
    4. Removing unnecessary spaces around parentheses, commas, and operators
    """
    if not sql:
        return ""
    # Lowercase
    sql = sql.lower()
    # Strip semicolons
    sql = sql.strip().strip(";")
    # Collapse all whitespace to single spaces
    sql = re.sub(r"\s+", " ", sql)
    # Remove spaces around operators and punctuations
    sql = re.sub(r"\s*([,()=><!+*/-])\s*", r"\1", sql)
    return sql.strip()

def validate_sql(schema: str, query: str) -> tuple[bool, str]:
    """
    Validates if a SQL query is syntactically correct against a schema by executing
    it on an in-memory SQLite database populated with dummy data.
    Returns (is_valid, error_message).
    """
    if not query:
        return False, "Empty query"
    
    # Strip potential markdown formatting
    if "```" in query:
        query = query.replace("```sql", "").replace("```", "").strip()
        
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON;")
        
        # Build the tables from the schema DDL
        cursor.executescript(schema)
        
        # Populate tables with dummy rows to prevent execution errors on empty tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        for table in tables:
            cursor.execute(f"PRAGMA table_info(\"{table}\");")
            columns = cursor.fetchall()
            
            col_names = []
            col_values = []
            for col in columns:
                col_name = col[1]
                col_type = col[2].upper()
                
                col_names.append(f'"{col_name}"')
                if "INT" in col_type:
                    col_values.append(1)
                elif "DECIMAL" in col_type or "NUMERIC" in col_type or "REAL" in col_type or "DOUBLE" in col_type or "FLOAT" in col_type:
                    col_values.append(10.0)
                elif "DATE" in col_type or "TIME" in col_type or "TIMESTAMP" in col_type:
                    col_values.append("'2026-08-10'")
                else:
                    col_values.append("'test_val'")
                    
            if col_names:
                insert_sql = f"INSERT OR IGNORE INTO \"{table}\" ({', '.join(col_names)}) VALUES ({', '.join(map(str, col_values))});"
                try:
                    cursor.execute(insert_sql)
                except Exception:
                    pass
                    
        conn.commit()
        
        # Execute query to verify syntax and schema correctness
        cursor.execute(query)
        cursor.fetchall()
        
        return True, ""
    except sqlite3.Error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)
    finally:
        try:
            conn.close()
        except NameError:
            pass

def main():
    print("[Eval] Initializing local ONNX SLMTextToSQL model...")
    try:
        agent = SLMTextToSQL(n_ctx=2048)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        sys.exit(1)

    print("\n[Eval] Loading dataset 'trl-lab/SQaLe-text-to-SQL-dataset' from Hugging Face...")
    try:
        dataset = load_dataset("trl-lab/SQaLe-text-to-SQL-dataset", split="train")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset: {e}")
        sys.exit(1)

    # Run only the failed samples (indices 50000, 50001, 50004)
    test_slice_indices = [50000, 50001, 50004]
    test_slice = dataset.select(test_slice_indices)

    print(f"\nEvaluating model on {len(test_slice_indices)} failed test samples (indices: {test_slice_indices})...\n")

    em_matches = 0
    valid_pred_count = 0
    test_slice_count = len(test_slice_indices)
    
    for idx, example in enumerate(test_slice):
        question = example["question"]
        schema = example["schema"]
        gold_query = example["query"]
        dataset_index = test_slice_indices[idx]
        
        print(f"--- Sample #{idx+1} (Dataset Index: {dataset_index}) ---")
        print(f"Question: {question}")
        
        try:
            pred_query = agent.generate_sql(schema=schema, question=question)
            print(f"GOLD: {gold_query}")
            print(f"PRED: {pred_query}")
            
            norm_gold = normalize_sql(gold_query)
            norm_pred = normalize_sql(pred_query)
            
            is_match = (norm_gold == norm_pred)
            print(f"Normalized Exact Match: {is_match}")
            
            # Syntax validation checks
            gold_valid, gold_err = validate_sql(schema, gold_query)
            pred_valid, pred_err = validate_sql(schema, pred_query)
            
            print(f"GOLD Execution Valid: {gold_valid} (Error: {gold_err if gold_err else 'None'})")
            print(f"PRED Execution Valid: {pred_valid} (Error: {pred_err if pred_err else 'None'})")
            
            if is_match:
                em_matches += 1
            if pred_valid:
                valid_pred_count += 1
                    
        except Exception as e:
            print(f"Generation failed: {e}")
            
        print()

    em_accuracy = (em_matches / test_slice_count) * 100
    valid_percentage = (valid_pred_count / test_slice_count) * 100
    
    print("=" * 50)
    print(f"EVALUATION COMPLETE")
    print(f"Exact Match Accuracy: {em_matches}/{test_slice_count} ({em_accuracy:.2f}%)")
    print(f"Execution/Syntax Validity Rate: {valid_pred_count}/{test_slice_count} ({valid_percentage:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
