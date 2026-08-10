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

    def generate_sql(self, schema: str, question: str, temperature: float = 0.0, max_tokens: int = None, stream: bool = False):
        """
        Translates a natural language question into an SQL query based on the database schema.
        """
        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_TEXT_TO_SQL_MAX_TOKENS", 256))
            
        if stream:
            system_prompt = (
                "You are an expert SQL assistant. You MUST first think step-by-step about the database joins, "
                "column selections, or filters inside <thought>...</thought> tags, and then provide the final SQL query answer."
            )
        else:
            system_prompt = "You are an expert SQL assistant."
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Schema:\n{schema}\n\nQuestion: {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
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
        
        if stream:
            def token_generator():
                tokenizer_stream = self.tokenizer.create_stream()
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        yield tokenizer_stream.decode(new_tokens[0])
            return token_generator()
        else:
            output_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    output_tokens.append(int(new_tokens[0]))
                    
            return self.tokenizer.decode(output_tokens).strip()
