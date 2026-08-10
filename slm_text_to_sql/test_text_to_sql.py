import os
import sys

# Ensure local path is prioritized to import local slm_text_to_sql
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_text_to_sql import SLMTextToSQL

def main():
    print("[Test] Initializing SLMTextToSQL (this may take a few seconds to load)...")
    try:
        agent = SLMTextToSQL(n_ctx=2048)
    except Exception as e:
        print(f"[ERROR] Failed to initialize agent: {e}")
        print("Note: If the ONNX model cache or dependencies are not loaded/installed yet, this is expected.")
        print("Make sure onnxruntime-genai is installed and model files are downloaded.")
        sys.exit(1)
        
    schema = """
    CREATE TABLE Users (
        UserID INT PRIMARY KEY,
        Username VARCHAR(50) NOT NULL,
        Email VARCHAR(100) UNIQUE,
        CreatedAt TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE TABLE Orders (
        OrderID INT PRIMARY KEY,
        UserID INT,
        TotalAmount DECIMAL(10, 2),
        OrderDate DATE,
        FOREIGN KEY (UserID) REFERENCES Users(UserID)
    );
    """
    
    test_cases = [
        {
            "name": "Select all columns from Users table",
            "schema": schema,
            "question": "Show all user information",
            "check": lambda ans: "SELECT" in ans.upper() and "USERS" in ans.upper()
        },
        {
            "name": "Aggregate order total count",
            "schema": schema,
            "question": "How many orders were placed in total?",
            "check": lambda ans: "SELECT" in ans.upper() and "COUNT" in ans.upper() and "ORDERS" in ans.upper()
        },
        {
            "name": "Join Users and Orders",
            "schema": schema,
            "question": "Find the total amount spent by user 'john_doe'.",
            "check": lambda ans: "SELECT" in ans.upper() and "JOIN" in ans.upper() and "TOTALAMOUNT" in ans.upper()
        }
    ]
    
    print("\n" + "="*50)
    print("RUNNING TEXT-TO-SQL VERIFICATION SUITE")
    print("="*50)
    
    passed_count = 0
    for idx, case in enumerate(test_cases):
        print(f"\nTest #{idx+1}: {case['name']}")
        print(f"Question: {case['question']}")
        
        try:
            sql_query = agent.generate_sql(schema=case["schema"], question=case["question"])
            print(f"Generated SQL:\n{sql_query}")
            
            if case["check"](sql_query):
                print("Result: PASSED")
                passed_count += 1
            else:
                print("Result: FAILED (Verification check failed)")
        except Exception as e:
            print(f"Result: FAILED with exception: {e}")
            
    print("\n" + "="*50)
    print(f"VERIFICATION COMPLETE: {passed_count}/{len(test_cases)} PASSED")
    print("="*50)
    
    if passed_count == len(test_cases):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
