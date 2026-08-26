import os
import sys
import sqlite3
import yaml
import re

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_DB_MIGRATION_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMDBMigrator:
    """
    Analyzes legacy database schemas and generates zero-downtime, CPU-optimized SQL migrations,
    Alembic Python scripts, and modern SQLAlchemy ORM model definitions offline.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.model = None
        self.tokenizer = None
        self._lazy_init_onnx()

    def _lazy_init_onnx(self):
        try:
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if not main_mod or not hasattr(main_mod, "get_shared_onnx_genai"):
                try:
                    import importlib
                    main_mod = importlib.import_module("main")
                except Exception:
                    main_mod = None
            if main_mod and hasattr(main_mod, "get_shared_onnx_genai"):
                self.model, self.tokenizer = main_mod.get_shared_onnx_genai()
        except Exception:
            pass

    def generate_migration(self, from_schema: str = "", to_schema: str = "", query: str = "", dialect: str = "postgresql", system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs) -> str:
        """
        Generates production-ready zero-downtime Alembic migration scripts and SQLAlchemy models.
        """
        self._lazy_init_onnx()
        
        request_text = (query or user_input or "").strip()
        if not request_text:
            if from_schema or to_schema:
                request_text = f"Generate an Alembic database migration from initial schema:\n{from_schema}\nto target schema:\n{to_schema}\nTarget dialect: {dialect}"
            else:
                return "Please provide a schema diff, table definition, or migration requirement to generate the migration script."

        default_sys = (
            "You are an expert Database Infrastructure Engineer specialized in zero-downtime database migrations.\n"
            "Analyze the requested schema change and generate a complete, production-ready Alembic Python migration script following zero-downtime best practices "
            "(such as PostgreSQL CONCURRENTLY index creation, adding nullable/default columns safely without table locks, "
            "and atomic step separation).\n\n"
            "Structure your output with:\n"
            "1. An explanation of the zero-downtime migration strategy.\n"
            "2. Complete Alembic Python script (`upgrade()` and `downgrade()`) inside a ```python ``` block.\n"
            "3. Corresponding raw SQL DDL statements inside a ```sql ``` block.\n"
            "4. The updated SQLAlchemy ORM model class definition inside a ```python ``` block."
        )
        active_sys = system_prompt or default_sys

        if self.model is not None and self.tokenizer is not None and og is not None:
            prompt = (
                "<|im_start|>system\n"
                f"{active_sys}<|im_end|>\n"
                f"<|im_start|>user\n{request_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + 700, temperature=0.3)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                tokens_out = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        tok_id = int(new_tokens[0])
                        if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        tokens_out.append(tok_id)
                        if token_callback:
                            tok_str = self.tokenizer.decode([tok_id])
                            token_callback(tok_str)
                res_text = self.tokenizer.decode(tokens_out).strip()
                if "<|im_end|>" in res_text:
                    res_text = res_text.replace("<|im_end|>", "").strip()
                if res_text:
                    return res_text
            except Exception as e:
                print(f"[SLMDBMigrator] Generation error: {e}")
                return f"Error generating database migration script: {e}"

        return "Database migration model is currently initializing. Please try again in a few moments."
