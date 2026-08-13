import os
import yaml

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    """
    Searches for config.yaml in environment variables, CWD, parent dirs,
    and package installation directories.
    Returns a tuple of (config_dict, config_file_path).
    """
    config_paths = [
        os.environ.get("SLM_TEXT_TO_SQL_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception as e:
                print(f"[SLMTextToSQL] Warning: Failed to load config from {path}: {e}")
    return {}, ""

class SLMTextToSQL:
    """
    A CPU-optimized Text-to-SQL translation agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )
            
        n_threads = n_threads or int(os.environ.get("SLM_TEXT_TO_SQL_N_THREADS", 4))
        n_ctx     = n_ctx     or int(os.environ.get("SLM_TEXT_TO_SQL_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_TEXT_TO_SQL_CACHE_DIR")

        # Wire thread count to ONNX Runtime (must be set before model load)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)

        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMTextToSQL] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("text_to_sql")
        if not model_config:
            raise ValueError("models.text_to_sql configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.text_to_sql in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        # Check if genai_config.json exists recursively in config_path
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Model directory not found at {config_path} and auto-download parameters (repo_id) are missing in config.yaml")
            
        print(f"[SLMTextToSQL] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                print(f"[SLMTextToSQL] Resolved model directory containing genai_config.json: {root}")
                return root
                
        return config_path

    # System prompt updated with SQLite few-shot examples using non-conflicting schemas
    _SYSTEM_PROMPT = (
        "You are an expert SQL query writer. Follow these rules strictly:\n"
        "1. Only use tables and columns that are explicitly defined in the provided schema. Do not guess or assume column/table names.\n"
        "2. Use correlated subqueries or JOINs when a value must be derived from another table.\n"
        "3. Use IS NULL / IS NOT NULL for null checks, never != '' or = ''.\n"
        "4. Use the correct aggregation: SUM for totals, COUNT for row counts, AVG for averages.\n"
        "5. Write syntactically valid SQL: WHERE must come after all JOINs.\n"
        "6. Minimize joins: Only join tables that are strictly necessary to answer the question. Do not perform redundant joins.\n"
        "7. Map synonyms accurately: If the question mentions 'customer' or 'client', map it to the `users` table or corresponding foreign key table in the schema. Never invent a `customers` table if it is not in the schema DDL.\n"
        "8. Return ONLY the final SQL query with no additional explanation, thought tags, or markdown.\n\n"
        "### Examples\n\n"
        "Example 1:\n"
        "Schema:\n"
        "CREATE TABLE employees (emp_id INT, first_name VARCHAR(50), is_active INT);\n"
        "Question: How many active employees are there?\n"
        "Assistant:\n"
        "SELECT COUNT(*) FROM employees WHERE is_active = 1;\n\n"
        "Example 2:\n"
        "Schema:\n"
        "CREATE TABLE sales (sale_id INT, seller_id INT, revenue DECIMAL);\n"
        "CREATE TABLE sellers (seller_id INT, name VARCHAR(50));\n"
        "Question: Get the total sales revenue for seller 'John Doe'.\n"
        "Assistant:\n"
        "SELECT SUM(s.revenue) FROM sales s JOIN sellers sr ON s.seller_id = sr.seller_id WHERE sr.name = 'John Doe';\n\n"
        "Example 3:\n"
        "Schema:\n"
        "CREATE TABLE students (student_id INT, name VARCHAR(50));\n"
        "CREATE TABLE courses (course_id INT, title VARCHAR(50));\n"
        "CREATE TABLE enrollments (enrollment_id INT, student_id INT, course_id INT);\n"
        "Question: Get the titles of courses taken by student 'Alice'.\n"
        "Assistant:\n"
        "SELECT c.title FROM courses c JOIN enrollments e ON c.course_id = e.course_id JOIN students s ON e.student_id = s.student_id WHERE s.name = 'Alice';\n\n"
        "Example 4:\n"
        "Schema:\n"
        "CREATE TABLE web_requests (request_id INT, request_time TIMESTAMP, status_code INT);\n"
        "Question: How many bad requests (status code 500) were recorded in the last 24 hours?\n"
        "Assistant:\n"
        "SELECT COUNT(*) FROM web_requests WHERE status_code = 500 AND request_time >= datetime('now', '-24 hours');"
    )

    @staticmethod
    def _prune_schema(schema: str, question: str, max_tables: int = 8) -> str:
        """
        Prunes the database DDL schema to include at most `max_tables` tables relevant to the
        question and their directly connected tables via foreign keys.

        Args:
            schema: Full DDL schema string.
            question: Natural language question.
            max_tables: Maximum number of tables to include in pruned output. Default is 8.
        """
        if not schema:
            return ""
            
        import re
        statements = [s.strip() for s in schema.split(";") if s.strip()]
        table_ddls = {}
        relations = {}
        
        for stmt in statements:
            match = re.search(r'CREATE\s+TABLE\s+("?\w+"?)', stmt, re.IGNORECASE)
            if match:
                table_name = match.group(1).replace('"', '').strip().lower()
                table_ddls[table_name] = stmt + ";"
                
                fk_targets = re.findall(r'REFERENCES\s+("?\w+"?)', stmt, re.IGNORECASE)
                relations[table_name] = [t.replace('"', '').strip().lower() for t in fk_targets]
                
        question_tokens = set(re.findall(r'\w+', question.lower()))
        
        # Priority 1: Directly matched tables
        direct_matches = []
        for table_name in table_ddls:
            if table_name in question_tokens or any(table_name in token or token in table_name for token in question_tokens):
                direct_matches.append(table_name)
                
        # Priority 2: Column matched tables
        column_matches = []
        for table_name, ddl in table_ddls.items():
            if table_name in direct_matches:
                continue
            col_matches = re.findall(r'("?\w+"?)\s+(?:INT|VARCHAR|TEXT|DECIMAL|NUMERIC|REAL|DOUBLE|FLOAT|DATE|TIME|TIMESTAMP|BOOLEAN|CHAR)', ddl, re.IGNORECASE)
            columns = [c.replace('"', '').strip().lower() for c in col_matches]
            if any(col in question_tokens for col in columns):
                column_matches.append(table_name)
                
        matched_tables = direct_matches + column_matches
        
        # Build connections (bi-directional relationship map)
        connections = {t: set() for t in table_ddls}
        for table_name, targets in relations.items():
            for target in targets:
                if target in connections:
                    connections[table_name].add(target)
                    connections[target].add(table_name)
                    
        # Collect connected tables (Priority 3)
        connected_tables = []
        for table in matched_tables:
            if table in connections:
                for conn in connections[table]:
                    if conn not in matched_tables and conn not in connected_tables:
                        connected_tables.append(conn)
                        
        # Select tables up to caller-specified maximum
        selected_tables = []
        for t in matched_tables + connected_tables:
            if t not in selected_tables:
                selected_tables.append(t)
            if len(selected_tables) >= max_tables:
                break
                
        if not selected_tables:
            selected_tables = list(table_ddls.keys())[:max_tables]
            
        pruned_statements = []
        for stmt in statements:
            match = re.search(r'CREATE\s+TABLE\s+("?\w+"?)', stmt, re.IGNORECASE)
            if match:
                table_name = match.group(1).replace('"', '').strip().lower()
                if table_name in selected_tables:
                    pruned_statements.append(stmt + ";")
                    
        return "\n\n".join(pruned_statements)

    @staticmethod
    def _validate_sql(schema: str, query: str) -> tuple[bool, str]:
        """
        Validates the SQL query against an in-memory SQLite database populated with schema DDL.
        """
        if not query:
            return False, "Empty query"
            
        import sqlite3
        conn = None
        try:
            conn = sqlite3.connect(":memory:")
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.executescript(schema)
            
            # Populate tables with exactly 1 dummy row to satisfy FK/column checks
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
            
            cursor.execute(query)
            cursor.fetchall()
            return True, ""
        except sqlite3.Error as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)
        finally:
            if conn:
                conn.close()

    def generate_sql(
        self,
        schema: str,
        question: str,
        temperature: float = 0.0,
        max_tokens: int = None,
        stream: bool = False,
        column_descriptions: dict = None,
        few_shot_examples: list = None,
        system_prompt: str = None,
        max_iterations: int = 5,
        max_pruned_tables: int = 8,
    ):
        """
        Translates a natural language question into a SQL query, with optional agentic
        self-correction when stream=False.

        Args:
            schema: Full DDL schema string (CREATE TABLE statements).
            question: Natural language question to translate.
            temperature: Sampling temperature for generation. Default is 0.0 (deterministic).
            max_tokens: Maximum number of tokens to generate. Uses env var or 512 as fallback.
            stream: If True, returns a token streaming generator (bypasses self-correction).
            column_descriptions: Optional dict mapping "table.column" -> "description string".
                                  E.g. {"users.status": "Either 'active' or 'inactive'"}.
                                  Injected into the prompt as a column glossary.
            few_shot_examples: Optional list of dicts with keys 'schema', 'question', 'sql'.
                               Overrides the built-in hardcoded examples when provided.
                               E.g. [{"schema": "CREATE TABLE ...", "question": "...", "sql": "..."}]
            system_prompt: Optional full system prompt string to override the built-in one entirely.
                           When provided, built-in rules and few-shot examples are ignored.
            max_iterations: Maximum number of agentic self-correction retries on failure. Default 5.
            max_pruned_tables: Maximum number of tables passed to the model in the pruned schema.
                               Default 8. Reduce to 5 for faster inference on simpler schemas.
        """
        pruned_schema = self._prune_schema(schema, question, max_tables=max_pruned_tables)

        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_TEXT_TO_SQL_MAX_TOKENS", 512))

        # --- Build system prompt ---
        if system_prompt:
            # Caller-supplied system prompt takes full precedence
            active_system_prompt = system_prompt
        else:
            # Start from built-in rules
            active_system_prompt = self._SYSTEM_PROMPT

            # Append caller-supplied few-shot examples if provided
            if few_shot_examples:
                examples_str = "\n### Additional Examples\n\n"
                for i, ex in enumerate(few_shot_examples, start=1):
                    examples_str += f"Example {i}:\n"
                    if ex.get("schema"):
                        examples_str += f"Schema:\n{ex['schema']}\n"
                    examples_str += f"Question: {ex['question']}\nAssistant:\n{ex['sql']}\n\n"
                active_system_prompt += examples_str

        # --- Build column descriptions glossary block ---
        col_desc_block = ""
        if column_descriptions:
            col_desc_block = "\n### Column Descriptions\n"
            for col_key, desc in column_descriptions.items():
                col_desc_block += f"- {col_key}: {desc}\n"
            col_desc_block += "\n"

        prompt = (
            "<|im_start|>system\n"
            f"{active_system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"### Database Schema\n{pruned_schema}\n"
            f"{col_desc_block}"
            f"\n### Question\n{question}\n\n### SQL Query<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        def _generate(prompt_str, temp=0.0):
            input_tokens = self.tokenizer.encode(prompt_str)
            params = og.GeneratorParams(self.model)
            total_max_length = len(input_tokens) + max_tokens
            search_options = {
                "max_length": total_max_length,
                "temperature": temp
            }
            params.set_search_options(**search_options)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            output_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    token_id = int(new_tokens[0])
                    if token_id in (151643, 151645):
                        break
                    output_tokens.append(token_id)
            
            raw_out = self.tokenizer.decode(output_tokens).strip()
            if "</thought>" in raw_out:
                raw_out = raw_out.split("</thought>")[-1].strip()
            elif "<thought>" in raw_out:
                raw_out = raw_out.split("<thought>")[0].strip()
            if "```" in raw_out:
                raw_out = raw_out.replace("```sql", "").replace("```", "").strip()
            return raw_out.strip()

        if stream:
            # Streaming bypasses self-correction
            input_tokens = self.tokenizer.encode(prompt)
            params = og.GeneratorParams(self.model)
            total_max_length = len(input_tokens) + max_tokens
            search_options = {
                "max_length": total_max_length,
                "temperature": temperature
            }
            params.set_search_options(**search_options)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            def token_generator():
                tokenizer_stream = self.tokenizer.create_stream()
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645):
                            break
                        yield tokenizer_stream.decode(token_id)
            return token_generator()

        # Non-streaming agentic self-correction loop
        query = _generate(prompt, temp=temperature)
        
        # Helper to extract table columns for all tables in the pruned schema to aid debugging
        def _get_pruned_table_schemas(pruned_schema_str):
            import re
            statements = [s.strip() for s in pruned_schema_str.split(";") if s.strip()]
            table_cols = {}
            for stmt in statements:
                match = re.search(r'CREATE\s+TABLE\s+("?\w+"?)', stmt, re.IGNORECASE)
                if match:
                    table_name = match.group(1).replace('"', '').strip().lower()
                    col_matches = re.findall(r'("?\w+"?)\s+(?:INT|INTEGER|VARCHAR|TEXT|DECIMAL|NUMERIC|REAL|DOUBLE|FLOAT|DATE|TIME|TIMESTAMP|BOOLEAN|CHAR)', stmt, re.IGNORECASE)
                    columns = [c.replace('"', '').strip() for c in col_matches]
                    table_cols[table_name] = columns
            schemas_feedback = []
            for t_name, cols in table_cols.items():
                schemas_feedback.append(f"- Table '{t_name}' columns: {', '.join(cols)}")
            return "\n".join(schemas_feedback)

        # Self-correction loop: up to max_iterations times
        failed_attempts = []
        for attempt in range(max_iterations):
            is_valid, err_msg = self._validate_sql(schema, query)
            if is_valid:
                break
                
            print(f"[Agentic Self-Correction] Attempt {attempt+1} failed database validation: {err_msg}")
            
            table_schemas = _get_pruned_table_schemas(pruned_schema)
            enhanced_error = err_msg
            if table_schemas:
                enhanced_error += f"\n\nAvailable column definitions for tables in your query:\n{table_schemas}"

            failed_attempts.append((query, enhanced_error))

            history_str = ""
            for idx, (failed_q, failed_err) in enumerate(failed_attempts):
                history_str += f"Failed Attempt #{idx+1} SQL:\n{failed_q}\nFailed Attempt #{idx+1} Database Error:\n{failed_err}\n\n"

            correction_prompt = (
                "<|im_start|>system\n"
                "You are an expert SQL query debugger. A previously generated SQL query failed to execute with a database error.\n"
                "Follow these rules strictly:\n"
                "1. Identify the cause of the database error and correct the SQL query.\n"
                "2. Only use tables and columns that are explicitly defined in the provided schema.\n"
                "3. Ensure all JOIN conditions align with the foreign key definitions in the DDL.\n"
                "4. Avoid ambiguous column names in SELECT clauses by always prefixing them with their table name or alias (e.g. use `users.id` instead of just `id`).\n"
                "5. Do NOT repeat any of the SQL queries that failed in the previous attempts.\n"
                "6. Return ONLY the corrected SQL query with no explanation, thought tags, or markdown.<|im_end|>\n"
                "<|im_start|>user\n"
                f"### Database Schema\n{pruned_schema}\n\n"
                f"### Question\n{question}\n\n"
                f"### Previously Failed Attempt(s) and Error(s)\n{history_str}"
                "### Corrected SQL Query<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            query = _generate(correction_prompt, temp=temperature)
            print(f"[Agentic Self-Correction] Generated corrected query: {query}")

        return query
