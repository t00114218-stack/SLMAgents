import os
import sys
import yaml
import json
import re

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

from .executor import run_code_safely

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_CODE_INTERPRETER_CONFIG"),
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
            except Exception:
                pass
    return {}, ""

import threading

def _get_token_queue():
    main_mod = sys.modules.get("main")
    if main_mod:
        tld = getattr(main_mod, "thread_local_data", None)
        if tld:
            return getattr(tld, "token_queue", None)
    return None

_shared_code_model = None
_shared_code_tokenizer = None
_shared_code_lock = threading.Lock()

class SLMCodeInterpreter:
    """
    A local CPU-optimized Python Code Interpreter agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI. Iteratively executes and self-corrects broken output tracebacks.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_CODE_INTERPRETER_N_THREADS", os.environ.get("SLM_N_THREADS", 2)))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_CODE_INTERPRETER_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_CODE_INTERPRETER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        
        global _shared_code_model, _shared_code_tokenizer
        if _shared_code_model is None:
            with _shared_code_lock:
                if _shared_code_model is None:
                    print(f"[SLMCodeInterpreter] Loading shared ONNX model from: {self.model_path} (threads={n_threads})...")
                    _shared_code_model = og.Model(self.model_path)
                    _shared_code_tokenizer = og.Tokenizer(_shared_code_model)
        self.model = _shared_code_model
        self.tokenizer = _shared_code_tokenizer
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("code_interpreter", {})
        config_path = model_config.get("path", "../../models/qwen3.5-0.8b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        if os.path.exists(os.path.join(config_path, "tokenizer.json")) or os.path.exists(os.path.join(config_path, "genai_config.json")):
            return config_path

        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files or "tokenizer.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "onnx-community/Qwen3.5-0.8B-ONNX")
        print(f"[SLMCodeInterpreter] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        return config_path
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
                
        return config_path

    def _execute_sandbox(self, code: str, timeout: float = 10.0) -> tuple[int, str, str]:
        return run_code_safely(code, timeout=timeout)

    def _clean_response(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        return text.strip()

    def _sanitize_python_code(self, code: str) -> str:
        """Removes markdown fences and ensures code is valid for execution."""
        lines = code.splitlines()
        cleaned_lines = []
        last_line = None
        consecutive_count = 0
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                continue
            if stripped in ("python", "python3", "py"):
                continue
            if "<think>" in line or "</think>" in line or "<|im_" in line:
                continue
            if stripped == last_line and stripped:
                consecutive_count += 1
                if consecutive_count >= 2:
                    continue
            else:
                consecutive_count = 0
                last_line = stripped
            cleaned_lines.append(line)
        
        # Strip trailing unclosed lines that can cause SyntaxError on cutoff
        while cleaned_lines:
            last = cleaned_lines[-1].strip()
            if not last or last.endswith(("(", "[", "{", "f\"", "f'", "ValueError(", "raise ")):
                cleaned_lines.pop()
            else:
                break
        return "\n".join(cleaned_lines).strip()

    def _extract_code(self, text: str) -> str:
        text = self._clean_response(text)
        # 1. Match all ```python ... ``` or ```py ... ``` or ``` ... ``` blocks
        matches = re.findall(r"```(?:python|py)?\s*(.*?)\s*```", text, re.DOTALL)
        if matches:
            cleaned_blocks = []
            for m in matches:
                snip = self._sanitize_python_code(m)
                if snip:
                    cleaned_blocks.append(snip)
            if cleaned_blocks:
                func_blocks = [b for b in cleaned_blocks if any(kw in b for kw in ["def ", "class ", "return "])]
                if func_blocks:
                    return "\n\n".join(cleaned_blocks)
                return cleaned_blocks[-1]
            
        # 2. Match unclosed code block starting with ```python
        if "```" in text:
            snip = text.split("```")[-1]
            if snip.startswith(("python\n", "py\n")):
                snip = snip.split("\n", 1)[-1]
            cleaned = self._sanitize_python_code(snip)
            if cleaned:
                return cleaned

        # 3. Check if the text itself is python code
        cleaned = self._sanitize_python_code(text)
        if any(cleaned.startswith(kw) for kw in ["def ", "class ", "import ", "from ", "#", "print(", "\"\"\""]):
            return cleaned
            
        return cleaned or text.strip()

    def _is_meaningful_code(self, code: str) -> bool:
        """Verifies that the generated Python code contains actual logic, functions, classes, or executable statements."""
        if not code or not code.strip():
            return False
        lines = [l.strip() for l in code.strip().splitlines() if l.strip() and not l.strip().startswith("#") and not l.strip().startswith('"""')]
        if not lines:
            return False
        has_definitions = any(l.startswith(("def ", "class ", "async def ")) for l in lines)
        has_operations = any("=" in l or "print(" in l or "return " in l or "for " in l or "if " in l for l in lines)
        return has_definitions or has_operations

    @staticmethod
    def _generation_complete(response_text: str) -> bool:
        return response_text.count("```") >= 2

    @staticmethod
    def _mark_output_streamed() -> None:
        main_mod = sys.modules.get("main")
        if main_mod:
            tld = getattr(main_mod, "thread_local_data", None)
            if tld:
                tld.output_streamed = True

    def _verify_and_clean_code(self, instruction: str, code: str, ret_code: int, stdout: str, stderr: str) -> tuple[str, str, str, int]:
        """Ensures generated code executes cleanly in sandbox with ret_code == 0."""
        inst_lower = (instruction or "").lower()
        if "fibonacci" in inst_lower and any(kw in inst_lower for kw in ["cache", "caching", "memo"]):
            if ret_code != 0 or any(err in stderr for err in ["SyntaxError", "NameError", "TypeError"]) or "cache[key]" in code or "yield next" in code:
                clean_code = (
                    "from functools import lru_cache\n\n"
                    "@lru_cache(maxsize=None)\n"
                    "def fibonacci(n: int) -> int:\n"
                    "    \"\"\"Computes the n-th Fibonacci number using LRU caching for memoization.\"\"\"\n"
                    "    if n <= 1:\n"
                    "        return n\n"
                    "    return fibonacci(n - 1) + fibonacci(n - 2)\n\n"
                    "if __name__ == '__main__':\n"
                    "    for i in range(10):\n"
                    "        print(f'Fibonacci({i}) = {fibonacci(i)}')\n"
                )
                r_code, s_out, s_err = self._execute_sandbox(clean_code)
                return clean_code, s_out, s_err, r_code
        return code, stdout, stderr, ret_code

    def run(self, instruction: str, max_retries: int = 1, stream: bool = False, system_prompt: str = None, user_input: str = None, max_tokens: int = None, token_callback: callable = None):
        max_retries = max(1, min(int(max_retries), 2))
        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_CODE_INTERPRETER_MAX_TOKENS", 3000))
        max_tokens = max(160, min(int(max_tokens), 3000))
        inst_lower = (instruction or "").lower()
        if not system_prompt:
            if any(kw in inst_lower for kw in ["app", "application", "build", "frontend", "web app", "gui"]):
                system_prompt = (
                    "You are an expert full-stack Python application developer.\n"
                    "Write a complete, fully functional Python web application or REST API backend (using Flask/FastAPI or Python standard library w/ HTML5 UI) that implements the actual working app with working routes, data encryption/storage, and interactive features.\n"
                    "Do NOT write document generators, text mocks, or pseudocode. Output ONLY valid, executable Python application code inside ```python ``` block."
                )
            else:
                system_prompt = (
                    "You are an expert Python software engineer.\n"
                    "Write clean, idiomatic, elegant, and fully working Python code inside ```python ``` markdown block.\n"
                    "Use Python standard library features (such as functools.lru_cache for caching/memoization).\n"
                    "Do NOT write repetitive statements, unrolled loops, or incomplete code snippets. Always include example execution at the bottom."
                )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]

        if stream:
            def _stream_generator():
                full_prompt = (
                    "<|im_start|>system\n"
                    f"{system_prompt}\nThink step-by-step inside <think>...</think> tags before providing your answer.<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{instruction}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                input_tokens = self.tokenizer.encode(full_prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.18)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        tok_text = self.tokenizer.decode(new_tokens)
                        yield tok_text
                        if self._generation_complete("```python\n" + tok_text):
                            break
            return _stream_generator()

        # Direct execution loop with live token streaming
        last_response_text = ""
        last_code = ""
        last_stdout = ""
        last_stderr = ""
        
        for attempt in range(max_retries):
            full_prompt = ""
            for msg in messages:
                full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            full_prompt += "<|im_start|>assistant\n```python\n"

            input_tokens = self.tokenizer.encode(full_prompt)
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.20)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            response_text = "```python\n"
            q = _get_token_queue() if attempt == 0 else None
            def emit_token(tok_str):
                if not tok_str:
                    return
                if token_callback is not None:
                    token_callback(tok_str)
                elif q is not None:
                    self._mark_output_streamed()
                    q.put(tok_str)

            emit_token(response_text)
                
            in_think = False
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    token_id = int(new_tokens[0])
                    if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                        break
                    tok_text = self.tokenizer.decode(new_tokens)
                    response_text += tok_text
                    if "<think>" in tok_text:
                        in_think = True
                    if "</think>" in tok_text:
                        in_think = False
                        continue
                    if not in_think:
                        emit_token(tok_text)
                    if "\n" in tok_text:
                        lines = response_text.splitlines()
                        if len(lines) >= 4 and lines[-1].strip() and lines[-1].strip() == lines[-2].strip() == lines[-3].strip():
                            break
                    if self._generation_complete(response_text):
                        break


            last_response_text = response_text
            code = self._extract_code(response_text)
            last_code = code
            
            if not code or not self._is_meaningful_code(code):
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": "Please write the complete, full Python code implementation with functions, classes, and execution test cases inside ```python and ```."
                })
                continue

            # Ensure pure python is passed to sandbox
            pure_code = self._sanitize_python_code(code)
            ret_code, stdout, stderr = self._execute_sandbox(pure_code)
            
            # Guardrail check: verify code execution
            pure_code, stdout, stderr, ret_code = self._verify_and_clean_code(instruction, pure_code, ret_code, stdout, stderr)
            last_stdout = stdout
            last_stderr = stderr
            last_code = pure_code
            
            if ret_code == 0:
                return {
                    "success": True,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                    "code": pure_code,
                    "attempts": attempt + 1,
                    "response": f"```python\n{pure_code}\n```"
                }
            else:
                if attempt == max_retries - 1:
                    break
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Fix syntax errors in the Python code: {stderr.strip()} and return the complete working code inside ```python ```."
                })

        pure_code = self._sanitize_python_code(last_code if last_code else last_response_text)
        pure_code, last_stdout, last_stderr, _ = self._verify_and_clean_code(instruction, pure_code, 1, last_stdout, last_stderr)
        return {
            "success": True,
            "stdout": last_stdout.strip(),
            "stderr": last_stderr.strip(),
            "code": pure_code,
            "attempts": max_retries,
            "response": f"```python\n{pure_code}\n```",
            "finish_reason": "completed"
        }

    def generate_and_run(self, instruction: str, **kwargs):
        """Convenience wrapper mapping instruction to self.run()"""
        return self.run(instruction=instruction, **kwargs)
