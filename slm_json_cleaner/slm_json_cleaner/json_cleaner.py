import os
import sys
import yaml
import json
import re

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_JSON_CLEANER_CONFIG"),
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

class SLMJSONCleaner:
    """
    A local CPU-optimized JSON text sanitizer powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI. Repair and format unstructured, broken strings into schema valid JSON structures.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_JSON_CLEANER_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_JSON_CLEANER_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_JSON_CLEANER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMJSONCleaner] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("json_cleaner", {})
        config_path = model_config.get("path", "../../models/qwen2.5-1.5b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX")
        print(f"[SLMJSONCleaner] ONNX Model not found at configured path. Auto-downloading...")
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

    def _extract_json_block(self, text: str) -> str:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        # Brace matcher fallback
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
            
        return text.strip()

    def clean_json(self, malformed_text: str, schema_dict: dict, stream: bool = False, system_prompt: str = None, user_input: str = None):
        """
        Sanitizes raw broken JSON input strings to match a schema schema_dict.
        If stream=True, returns a token generator.
        If stream=False, runs to completion and parses the cleaned structured response.
        """
        system_prompt = (
            "You are a local JSON sanitization utility.\n"
            "Analyze the malformed unstructured text, think inside <thought>...</thought> tags, "
            "and output the sanitized repaired JSON block complying with the target schema inside a ```json ... ``` code block. "
            "Never append explanations outside the code block."
        )

        user_prompt = (
            f"Broken Text:\n{malformed_text}\n\n"
            f"Required JSON Structure Schema:\n{json.dumps(schema_dict, indent=2)}"
        )

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_tokens = self.tokenizer.encode(full_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + 512, temperature=0.0)

        if stream:
            def _stream_generator():
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645):
                            break
                        yield self.tokenizer.decode(new_tokens)
            return _stream_generator()

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        response_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                token_id = int(new_tokens[0])
                if token_id in (151643, 151645):
                    break
                response_text += self.tokenizer.decode(new_tokens)

        json_block = self._extract_json_block(response_text)
        try:
            parsed = json.loads(json_block)
            return parsed, True
        except Exception:
            return {"raw_output": json_block, "error": "Standard JSON decoding failed"}, False
