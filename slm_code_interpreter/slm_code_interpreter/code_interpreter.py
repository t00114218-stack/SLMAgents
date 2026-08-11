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

        n_threads = n_threads or int(os.environ.get("SLM_CODE_INTERPRETER_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_CODE_INTERPRETER_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_CODE_INTERPRETER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMCodeInterpreter] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("code_interpreter", {})
        config_path = model_config.get("path", "../../models/qwen2.5-1.5b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX")
        print(f"[SLMCodeInterpreter] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
                
        return config_path

    def _execute_sandbox(self, code: str, timeout: float = 10.0) -> tuple[int, str, str]:
        return run_code_safely(code, timeout=timeout)

    def _extract_code(self, text: str) -> str:
        match = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            code = match.group(1).strip()
            lines = code.splitlines()
            if lines and lines[0].strip().lower() in ("python", "python3", "py"):
                code = "\n".join(lines[1:]).strip()
            return code
        return ""


    def run(self, instruction: str, max_retries: int = 3, stream: bool = False):
        """
        Executes code instructions. If stream=True, returns a generator of tokens.
        If stream=False, performs self-correcting run to completion and returns the dict response.
        """
        system_prompt = (
            "You are a local Python interpreter agent.\n"
            "Analyze the instruction, think inside <thought>...</thought> tags, and then write "
            "executable Python code wrapped inside a ```python ... ``` code block. "
            "Ensure you write robust code and print outputs clearly. Never use interactive inputs."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": instruction}
        ]

        if stream:
            def _stream_generator():
                full_prompt = (
                    "<|im_start|>system\n"
                    f"{system_prompt}<|im_end|>\n"
                    "<|im_start|>user\n"
                    f"{instruction}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                input_tokens = self.tokenizer.encode(full_prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + 1024, temperature=0.0)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        yield self.tokenizer.decode(new_tokens)
            return _stream_generator()

        # Self-correction execution loop when stream=False
        for attempt in range(max_retries):
            full_prompt = ""
            for msg in messages:
                full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            full_prompt += "<|im_start|>assistant\n"

            input_tokens = self.tokenizer.encode(full_prompt)
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=len(input_tokens) + 1024, temperature=0.0)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            response_text = ""
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    response_text += self.tokenizer.decode(new_tokens)

            code = self._extract_code(response_text)
            if not code:
                # Prompt again for code format
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": "You forgot to output the code wrapped inside a ```python and ``` code block. Try again."})
                continue

            ret_code, stdout, stderr = self._execute_sandbox(code)
            if ret_code == 0:
                return {
                    "success": True,
                    "stdout": stdout.strip(),
                    "stderr": stderr.strip(),
                    "code": code,
                    "attempts": attempt + 1,
                    "response": response_text
                }
            else:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"The code execution failed with return code {ret_code}.\nError logs:\n{stderr}\nCorrect your code errors and return the complete updated code inside ```python ```."
                })

        return {
            "success": False,
            "stdout": "",
            "stderr": "Max correction loops reached.",
            "code": "",
            "attempts": max_retries,
            "response": ""
        }
