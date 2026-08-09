import os
import sys
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
        os.environ.get("SLM_RAG_CONFIG"),
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
                raise ValueError(f"Failed to parse config file at {path}: {e}")
    raise FileNotFoundError("config.yaml not found in environment, current directory, or package directories.")

class SLMRag:
    """
    A CPU-optimized Retrieval-Augmented Generation (RAG) runner powered by a local
    Small Language Model (SLM) running via ONNX Runtime GenAI.
    Answers user questions based on provided document chunks while strictly adhering
    to user instructions.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=8192, n_threads=4):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )
            
        # Resolve the ONNX model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMRag] Loading ONNX model from: {self.model_path}...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
            
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        """
        Locates or downloads the necessary ONNX model as defined in config.yaml.
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        # Check config.yaml
        config = load_config()
        model_config = config.get("models", {}).get("rag")
        if not model_config:
            raise ValueError("models.rag configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.rag in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        
        # Check if genai_config.json exists recursively in config_path
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Model file not found at {config_path} and auto-download parameters (repo_id) are missing in config.yaml")
            
        print(f"[SLMRag] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        # Scan again to find actual directory containing genai_config.json
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                print(f"[SLMRag] Resolved model directory containing genai_config.json: {root}")
                return root
                
        return config_path

    def answer(self, chunks: list, question: str, instruction: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Synthesizes an answer based on document chunks, user question, and user instruction.
        """
        # Format the text chunks for context
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"--- Chunk {i+1} ---\n{chunk.strip()}\n\n"
            
        # Build strict ChatML template prompt
        system_prompt = (
            "You are a precise and helpful assistant. Your task is to answer the user's question "
            "based ONLY on the provided text chunks. If the chunks do not contain the answer, say "
            "so clearly. You must strictly adhere to the instruction provided by the user."
        )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}\n\n"
            f"Instruction to follow: {instruction}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Text Chunks:\n{formatted_chunks}"
            f"User Question: {question}\n\n"
            f"Remember, you must adhere to the instruction: {instruction}<|im_end|>\n"
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
