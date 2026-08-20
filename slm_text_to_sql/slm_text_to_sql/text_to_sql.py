import os
import sys
import re
import yaml
import threading

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

def _get_token_queue():
    main_mod = sys.modules.get("main")
    if main_mod:
        tld = getattr(main_mod, "thread_local_data", None)
        if tld:
            return getattr(tld, "token_queue", None)
    return None

_shared_sql_model = None
_shared_sql_tokenizer = None
_shared_sql_lock = threading.Lock()

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
            
        n_threads = n_threads or int(os.environ.get("SLM_TEXT_TO_SQL_N_THREADS", os.environ.get("SLM_N_THREADS", 2)))
        n_ctx     = n_ctx     or int(os.environ.get("SLM_TEXT_TO_SQL_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_TEXT_TO_SQL_CACHE_DIR")

        # Wire thread count to ONNX Runtime (must be set before model load)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)

        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        global _shared_sql_model, _shared_sql_tokenizer
        if _shared_sql_model is None:
            with _shared_sql_lock:
                if _shared_sql_model is None:
                    print(f"[SLMTextToSQL] Loading shared ONNX model from: {self.model_path} (threads={n_threads})...")
                    _shared_sql_model = og.Model(self.model_path)
                    _shared_sql_tokenizer = og.Tokenizer(_shared_sql_model)
        self.model = _shared_sql_model
        self.tokenizer = _shared_sql_tokenizer

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
        "1. Strictly use ONLY tables and columns that are explicitly defined in the provided DDL schema. NEVER invent, hallucinate, or guess table/column names (e.g. do NOT output 'users' unless 'CREATE TABLE users' is in the schema DDL).\n"
        "2. Use correlated subqueries or JOINs when a value must be derived from another table.\n"
        "3. Use IS NULL / IS NOT NULL for null checks, never != '' or = ''.\n"
        "4. Use the correct aggregation: SUM for totals, COUNT for row counts, AVG for averages.\n"
        "5. CRITICAL SYNTAX RULE: ALL JOIN clauses (INNER JOIN, LEFT JOIN) MUST come BEFORE the WHERE clause! NEVER write a WHERE clause before a JOIN clause.\n"
        "6. Minimize joins: Only join tables that are strictly necessary to answer the question. Do not perform redundant joins.\n"
        "7. Strict Table Adherence: If a question mentions 'users', 'customers', or 'clients', NEVER write `JOIN users` or `FROM users` UNLESS `CREATE TABLE users` (or `users_roles` etc.) is explicitly defined in the provided DDL schema. Use existing ID columns (e.g. `receiver_id`, `user_id_a`) directly.\n"
        "8. Fully complete all nested subqueries and close all opened parentheses before ending with a semicolon. Return ONLY the complete SQL query with no explanation, thought tags, or markdown."
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
            match = re.search(r'CREATE\s+TABLE\s+[`"]?([\w\-]+)[`"]?', stmt, re.IGNORECASE)
            if match:
                table_name = match.group(1).replace('`', '').replace('"', '').strip().lower()
                table_ddls[table_name] = stmt + ";"
                
                fk_targets = re.findall(r'REFERENCES\s+[`"]?([\w\-]+)[`"]?', stmt, re.IGNORECASE)
                relations[table_name] = [t.replace('`', '').replace('"', '').strip().lower() for t in fk_targets]
                
        # If total tables in schema is less than or equal to max_tables, return full schema
        if len(table_ddls) <= max_tables:
            return schema.strip()
            
        question_tokens = set(re.findall(r'\w+', question.lower()))
        # Add word stems (e.g. 'licenses' -> 'license', 'subscriptions' -> 'subscription')
        stems = set()
        for tok in question_tokens:
            stems.add(tok)
            if tok.endswith('s') and len(tok) > 3:
                stems.add(tok[:-1])
            if tok.endswith('es') and len(tok) > 4:
                stems.add(tok[:-2])
            if tok.endswith('ies') and len(tok) > 4:
                stems.add(tok[:-3] + 'y')
                
        # Priority 1: Directly matched tables (matching stems and substrings)
        direct_matches = []
        for table_name in table_ddls:
            t_clean = table_name.lower()
            t_words = set(re.findall(r'\w+', t_clean))
            if any(st in t_clean or any(st in tw or tw in st for tw in t_words) for st in stems if len(st) >= 3):
                direct_matches.append(table_name)
                
        # Priority 2: Column matched tables
        column_matches = []
        for table_name, ddl in table_ddls.items():
            if table_name in direct_matches:
                continue
            # Extract column names from CREATE TABLE statement lines
            lines = ddl.split("\n")
            cols = []
            for l in lines:
                m_col = re.search(r'^\s*[`"]?([\w\-]+)[`"]?', l)
                if m_col:
                    c_name = m_col.group(1).lower()
                    if c_name not in ('create', 'table', 'primary', 'foreign', 'key', 'constraint', 'unique', 'check', 'index'):
                        cols.append(c_name)
            if any(any(st in c for c in cols) for st in stems if len(st) >= 3):
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
            match = re.search(r'CREATE\s+TABLE\s+[`"]?([\w\-]+)[`"]?', stmt, re.IGNORECASE)
            if match:
                table_name = match.group(1).replace('`', '').replace('"', '').strip().lower()
                if table_name in selected_tables:
                    pruned_statements.append(stmt + ";")
                    
        return "\n\n".join(pruned_statements)

    @staticmethod
    def _validate_sql(schema: str, query: str) -> tuple[bool, str]:
        """
        Validates the SQL query against an in-memory SQLite database populated with schema DDL.
        """
        if not query or not query.strip():
            return False, "Empty query"
        if not schema or not schema.strip():
            return True, ""
            
        import sqlite3
        conn = None
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
            
            try:
                cursor.execute(query)
                cursor.fetchall()
                return True, ""
            except sqlite3.OperationalError as op_err:
                err_str = str(op_err)
                return False, err_str
        except sqlite3.Error as e:
            return False, str(e)
        except Exception as e:
            return True, ""
        finally:
            if conn:
                conn.close()

    @staticmethod
    def _extract_fk_relationships(schema_str: str) -> str:
        if not schema_str:
            return ""
        import re
        lines = schema_str.split("\n")
        fk_lines = []
        current_table = None
        for line in lines:
            m_tab = re.search(r'CREATE\s+TABLE\s+[`"]?([\w\-]+)[`"]?', line, re.IGNORECASE)
            if m_tab:
                current_table = m_tab.group(1).replace('`', '').replace('"', '').strip()
            m_fk = re.search(r'FOREIGN\s+KEY\s*\(([^)]+)\)\s*REFERENCES\s+[`"]?([\w\-]+)[`"]?\s*\(([^)]+)\)', line, re.IGNORECASE)
            if m_fk and current_table:
                src_cols = m_fk.group(1).replace('`', '').replace('"', '').strip()
                target_table = m_fk.group(2).replace('`', '').replace('"', '').strip()
                target_cols = m_fk.group(3).replace('`', '').replace('"', '').strip()
                fk_lines.append(f"- Table `{current_table}` ({src_cols}) -> `{target_table}` ({target_cols})")
        if not fk_lines:
            return ""
        return "\n### Foreign Key Relationships & Table Links\n" + "\n".join(fk_lines[:20]) + "\n"

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
        """
        pruned_schema = self._prune_schema(schema, question, max_tables=max_pruned_tables)
        fk_block = self._extract_fk_relationships(pruned_schema)

        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_TEXT_TO_SQL_MAX_TOKENS", 512))

        # --- Build system prompt ---
        if system_prompt:
            active_system_prompt = system_prompt
        else:
            active_system_prompt = self._SYSTEM_PROMPT
            if few_shot_examples:
                examples_str = "\n### Additional Examples\n\n"
                for i, ex in enumerate(few_shot_examples, start=1):
                    examples_str += f"Example {i}:\n"
                    if ex.get("schema"):
                        examples_str += f"Schema:\n{ex['schema']}\n"
                    examples_str += f"Question: {ex['question']}\nAssistant:\n{ex['sql']}\n\n"
                active_system_prompt += examples_str

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
            f"{fk_block}"
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
            accumulated = ""
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    token_id = int(new_tokens[0])
                    if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                        break
                    output_tokens.append(token_id)
                    tok_text = self.tokenizer.decode(new_tokens)
                    accumulated += tok_text
                    # Break loop if repetitive subqueries or lines occur
                    if accumulated.count("IN (") >= 3 or accumulated.count("SELECT") >= 4:
                        break
                    if ";" in tok_text and len(output_tokens) > 5:
                        break
            
            raw_out = self.tokenizer.decode(output_tokens).strip()
            raw_out = re.sub(r'<think>.*?</think>', '', raw_out, flags=re.DOTALL).strip()
            raw_out = re.sub(r'</?think>', '', raw_out).strip()
            if "</thought>" in raw_out:
                raw_out = raw_out.split("</thought>")[-1].strip()
            if "```" in raw_out:
                raw_out = raw_out.replace("```sql", "").replace("```", "").strip()
            # If subqueries looped, clean up trailing unclosed parentheses
            if raw_out.count("(") > raw_out.count(")"):
                raw_out = raw_out.split("IN (")[0].strip()
                if not raw_out.endswith(";"):
                    raw_out += ";"
            return raw_out.strip()

        if stream:
            input_tokens = self.tokenizer.encode(prompt)
            params = og.GeneratorParams(self.model)
            total_max_length = len(input_tokens) + max_tokens
            search_options = {
                "max_length": total_max_length,
                "temperature": temperature,
                "repetition_penalty": 1.15
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
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        yield tokenizer_stream.decode(token_id)
            return token_generator()

        # Non-streaming self-correction loop
        query = _generate(prompt, temp=temperature)
        if not schema or not schema.strip():
            return query
        
        def _get_pruned_table_schemas(pruned_schema_str):
            statements = [s.strip() for s in pruned_schema_str.split(";") if s.strip()]
            table_cols = {}
            for stmt in statements:
                match = re.search(r'CREATE\s+TABLE\s+[`"]?([\w\-]+)[`"]?', stmt, re.IGNORECASE)
                if match:
                    table_name = match.group(1).replace('`', '').replace('"', '').strip().lower()
                    col_matches = re.findall(r'[`"]?([\w\-]+)[`"]?\s+(?:INT|INTEGER|VARCHAR|TEXT|DECIMAL|NUMERIC|REAL|DOUBLE|FLOAT|DATE|TIME|TIMESTAMP|BOOLEAN|CHAR)', stmt, re.IGNORECASE)
                    columns = [c.replace('`', '').replace('"', '').strip() for c in col_matches]
                    table_cols[table_name] = columns
            schemas_feedback = []
            for t_name, cols in table_cols.items():
                if cols:
                    schemas_feedback.append(f"- Table '{t_name}' columns: {', '.join(cols)}")
                else:
                    schemas_feedback.append(f"- Table '{t_name}'")
            return "\n".join(schemas_feedback)

        max_iterations = max(1, min(int(max_iterations), 3))
        failed_attempts = []
        for attempt in range(max_iterations):
            is_valid, err_msg = self._validate_sql(schema, query)
            if is_valid:
                return query
                
            table_schemas = _get_pruned_table_schemas(pruned_schema)
            enhanced_error = err_msg
            if table_schemas:
                enhanced_error += f"\n\nAvailable column definitions for tables in your query:\n{table_schemas}"

            failed_attempts.append((query, enhanced_error))

            if attempt == max_iterations - 1:
                return query or f"-- Generated query for {question}"

            history_str = ""
            for idx, (failed_q, failed_err) in enumerate(failed_attempts[-2:]):
                history_str += f"Failed Attempt #{idx+1} SQL:\n{failed_q}\nFailed Attempt #{idx+1} Database Error:\n{failed_err}\n\n"

            correction_prompt = (
                "<|im_start|>system\n"
                "You are an expert SQL query debugger. A previously generated SQL query failed to execute with a database error.\n"
                "Follow these rules strictly:\n"
                "1. Identify the cause of the database error and correct the SQL query.\n"
                "2. Only use tables and columns that are explicitly defined in the provided schema.\n"
                "3. Ensure all JOIN conditions align with the foreign key definitions in the DDL.\n"
                "4. Return ONLY the corrected SQL query with no explanation, thought tags, or markdown.<|im_end|>\n"
                "<|im_start|>user\n"
                f"### Database Schema\n{pruned_schema}\n\n"
                f"### Question\n{question}\n\n"
                f"### Previously Failed Attempt(s) and Error(s)\n{history_str}"
                "### Corrected SQL Query<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            query = _generate(correction_prompt, temp=temperature)

        return query
