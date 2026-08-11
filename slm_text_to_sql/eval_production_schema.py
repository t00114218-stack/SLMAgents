import os
import sys
import re
import sqlite3

# Prioritize local path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_text_to_sql import SLMTextToSQL

# 1. Define the 15-table E-commerce schema DDL
PRODUCTION_SCHEMA = """
CREATE TABLE users ( -- Represents users and customers
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    role TEXT DEFAULT 'customer',
    status TEXT DEFAULT 'active'
);
CREATE TABLE user_profiles (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    theme TEXT DEFAULT 'light',
    language TEXT DEFAULT 'en'
);
CREATE TABLE user_login_attempts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    success_status INTEGER NOT NULL,
    attempt_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE devices (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    platform TEXT,
    platform_version TEXT
);
CREATE TABLE categories (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE products ( -- Represents products. Link to orders via order_items.
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    stock INTEGER DEFAULT 0,
    category_id INTEGER REFERENCES categories(id)
);
CREATE TABLE orders ( -- Represents orders. Link to products via order_items.
    id INTEGER PRIMARY KEY,
    customer_id INTEGER REFERENCES users(id),
    total_amount REAL NOT NULL,
    order_date DATE DEFAULT CURRENT_DATE,
    status TEXT DEFAULT 'pending'
);
CREATE TABLE order_items (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL,
    price REAL NOT NULL
);
CREATE TABLE payments (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    amount REAL NOT NULL,
    payment_method TEXT,
    payment_status TEXT,
    payment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE shipments (
    id INTEGER PRIMARY KEY,
    order_id INTEGER REFERENCES orders(id),
    tracking_number TEXT,
    status TEXT DEFAULT 'pending',
    shipped_date TIMESTAMP
);
CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    user_id INTEGER REFERENCES users(id),
    rating INTEGER CHECK(rating >= 1 AND rating <= 5),
    comment TEXT,
    review_date DATE DEFAULT CURRENT_DATE
);
CREATE TABLE cart (
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE cart_items (
    id INTEGER PRIMARY KEY,
    cart_id INTEGER REFERENCES cart(id),
    product_id INTEGER REFERENCES products(id),
    quantity INTEGER NOT NULL
);
CREATE TABLE coupons (
    id INTEGER PRIMARY KEY,
    code TEXT UNIQUE NOT NULL,
    discount_percent INTEGER NOT NULL,
    active INTEGER DEFAULT 1
);
CREATE TABLE support_tickets ( -- Represents support tickets and user tickets
    id INTEGER PRIMARY KEY,
    user_id INTEGER REFERENCES users(id),
    subject TEXT,
    status TEXT DEFAULT 'open',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# 2. Define 20 evaluation test cases specifically targeting this schema
EVALUATION_CASES = [
    {
        "id": 1,
        "question": "List all users who have an active status.",
        "gold": "SELECT * FROM users WHERE status = 'active';"
    },
    {
        "id": 2,
        "question": "How many products are in stock for each category?",
        "gold": "SELECT category_id, SUM(stock) FROM products GROUP BY category_id;"
    },
    {
        "id": 3,
        "question": "Find the names of users who placed order number 5.",
        "gold": "SELECT u.name FROM users u JOIN orders o ON u.id = o.customer_id WHERE o.id = 5;"
    },
    {
        "id": 4,
        "question": "Calculate the total amount spent by customer 'John Doe'.",
        "gold": "SELECT SUM(o.total_amount) FROM orders o JOIN users u ON o.customer_id = u.id WHERE u.name = 'John Doe';"
    },
    {
        "id": 5,
        "question": "Show the average review rating for the product named 'Laptop'.",
        "gold": "SELECT AVG(r.rating) FROM reviews r JOIN products p ON r.product_id = p.id WHERE p.name = 'Laptop';"
    },
    {
        "id": 6,
        "question": "How many successful logins happened in the last 30 days?",
        "gold": "SELECT COUNT(*) FROM user_login_attempts WHERE success_status = 1 AND attempt_time >= datetime('now', '-30 days');"
    },
    {
        "id": 7,
        "question": "Get support tickets subject for users who have a 'dark' theme preference.",
        "gold": "SELECT s.subject FROM support_tickets s JOIN users u ON s.user_id = u.id JOIN user_profiles p ON u.id = p.user_id WHERE p.theme = 'dark';"
    },
    {
        "id": 8,
        "question": "List products that were ordered by users using an 'Android' device.",
        "gold": "SELECT DISTINCT p.name FROM products p JOIN order_items oi ON p.id = oi.product_id JOIN orders o ON oi.order_id = o.id JOIN devices d ON o.customer_id = d.user_id WHERE d.platform = 'Android';"
    },
    {
        "id": 9,
        "question": "Find the total quantity of cart items in user 'Jane Smith's cart.",
        "gold": "SELECT SUM(ci.quantity) FROM cart_items ci JOIN cart c ON ci.cart_id = c.id JOIN users u ON c.user_id = u.id WHERE u.name = 'Jane Smith';"
    },
    {
        "id": 10,
        "question": "Get all active coupons with a discount percentage greater than 20.",
        "gold": "SELECT * FROM coupons WHERE active = 1 AND discount_percent > 20;"
    },
    {
        "id": 11,
        "question": "Find the tracking number for the shipment of order number 3.",
        "gold": "SELECT tracking_number FROM shipments WHERE order_id = 3;"
    },
    {
        "id": 12,
        "question": "List all orders with a status of 'delivered'.",
        "gold": "SELECT * FROM orders WHERE status = 'delivered';"
    },
    {
        "id": 13,
        "question": "Get the email addresses of users who have never placed an order.",
        "gold": "SELECT u.email FROM users u WHERE u.id NOT IN (SELECT customer_id FROM orders);"
    },
    {
        "id": 14,
        "question": "Count the number of reviews for each product.",
        "gold": "SELECT product_id, COUNT(*) AS review_count FROM reviews GROUP BY product_id;"
    },
    {
        "id": 15,
        "question": "Find all payments made using 'credit_card'.",
        "gold": "SELECT * FROM payments WHERE payment_method = 'credit_card';"
    },
    {
        "id": 16,
        "question": "List the names and prices of products that cost more than 100.",
        "gold": "SELECT name, price FROM products WHERE price > 100;"
    },
    {
        "id": 17,
        "question": "How many devices does each user have?",
        "gold": "SELECT user_id, COUNT(*) AS device_count FROM devices GROUP BY user_id;"
    },
    {
        "id": 18,
        "question": "Get the top 5 most expensive products.",
        "gold": "SELECT name, price FROM products ORDER BY price DESC LIMIT 5;"
    },
    {
        "id": 19,
        "question": "Find all shipments that have not yet been shipped (status is pending).",
        "gold": "SELECT * FROM shipments WHERE status = 'pending';"
    },
    {
        "id": 20,
        "question": "Get the total revenue from all completed payments.",
        "gold": "SELECT SUM(amount) FROM payments WHERE payment_status = 'completed';"
    }
]

def normalize_sql(sql: str) -> str:
    if not sql:
        return ""
    sql = sql.lower().strip().strip(";")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*([,()=><!+*/-])\s*", r"\1", sql)
    return sql.strip()

def validate_sql(schema: str, query: str) -> tuple[bool, str]:
    if not query:
        return False, "Empty query"
    try:
        conn = sqlite3.connect(":memory:")
        cursor = conn.cursor()
        cursor.execute("PRAGMA foreign_keys = ON;")
        cursor.executescript(schema)
        
        # Populate tables with dummy rows
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
                elif "REAL" in col_type or "DECIMAL" in col_type or "NUMERIC" in col_type:
                    col_values.append(10.0)
                elif "DATE" in col_type or "TIME" in col_type or "TIMESTAMP" in col_type:
                    col_values.append("'2026-08-11'")
                else:
                    col_values.append("'test'")
            if col_names:
                insert_sql = f"INSERT OR IGNORE INTO \"{table}\" ({', '.join(col_names)}) VALUES ({', '.join(map(str, col_values))});"
                cursor.execute(insert_sql)
        conn.commit()
        
        # Execute test query
        cursor.execute(query)
        cursor.fetchall()
        return True, ""
    except sqlite3.Error as e:
        return False, str(e)
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()

def main():
    print("[Production Eval] Initializing local ONNX SLMTextToSQL model...")
    try:
        agent = SLMTextToSQL(n_ctx=2048)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        sys.exit(1)

    print(f"\nEvaluating model on {len(EVALUATION_CASES)} custom E-commerce production queries against a 15-table schema...\n")

    em_matches = 0
    valid_pred_count = 0
    total_cases = len(EVALUATION_CASES)

    for case in EVALUATION_CASES:
        qid = case["id"]
        question = case["question"]
        gold_query = case["gold"]

        print(f"--- Case #{qid} ---")
        print(f"Question: {question}")

        try:
            pred_query = agent.generate_sql(schema=PRODUCTION_SCHEMA, question=question)
            print(f"GOLD: {gold_query}")
            print(f"PRED: {pred_query}")

            norm_gold = normalize_sql(gold_query)
            norm_pred = normalize_sql(pred_query)

            is_match = (norm_gold == norm_pred)
            print(f"Exact Match: {is_match}")

            gold_valid, gold_err = validate_sql(PRODUCTION_SCHEMA, gold_query)
            pred_valid, pred_err = validate_sql(PRODUCTION_SCHEMA, pred_query)

            print(f"GOLD Execution Valid: {gold_valid} (Error: {gold_err if gold_err else 'None'})")
            print(f"PRED Execution Valid: {pred_valid} (Error: {pred_err if pred_err else 'None'})")

            if is_match:
                em_matches += 1
            if pred_valid:
                valid_pred_count += 1
        except Exception as e:
            print(f"Generation failed: {e}")
        print()

    em_accuracy = (em_matches / total_cases) * 100
    valid_percentage = (valid_pred_count / total_cases) * 100

    print("=" * 50)
    print("PRODUCTION EVALUATION COMPLETE")
    print(f"Exact Match Accuracy: {em_matches}/{total_cases} ({em_accuracy:.2f}%)")
    print(f"Execution/Syntax Validity Rate: {valid_pred_count}/{total_cases} ({valid_percentage:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
