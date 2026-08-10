import os
import yaml

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> dict:
    """
    Searches for config.yaml in environment variables, CWD, parent dirs,
    and package installation directories.
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
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[SLMTextToSQL] Warning: Failed to load config from {path}: {e}")
    return {}

class SLMTextToSQL:
    """
    A CPU-optimized Text-to-SQL translation agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=2048, n_threads=4):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMTextToSQL] Loading ONNX model from: {self.model_path}...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config = load_config()
        model_config = config.get("models", {}).get("text_to_sql")
        if not model_config:
            raise ValueError("models.text_to_sql configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.text_to_sql in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        
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

    def generate_sql(self, schema: str, question: str, temperature: float = 0.0, max_tokens: int = 256) -> str:
        """
        Translates a natural language question into an SQL query based on the database schema.
        """
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
        
        output_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                output_tokens.append(int(new_tokens[0]))
                
        return self.tokenizer.decode(output_tokens).strip()
