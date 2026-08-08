import os
import sys
from llama_cpp import Llama

def load_config() -> dict:
    """
    Searches for config.yaml in environment variables, CWD, parent dirs,
    and package installation directories.
    """
    try:
        import yaml
    except ImportError:
        return {}
        
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
    Small Language Model (SLM). It answers user questions based on provided document chunks
    while strictly adhering to user instructions.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=131072, n_threads=4):
        # Resolve the GGUF model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        
        print(f"[SLMRag] Loading model from: {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=True,
                verbose=False
            )
        except Exception as e:
            print(f"[SLMRag] Warning: Failed to load with use_mlock=True: {e}. Retrying without mlock...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=False,
                verbose=False
            )
            
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        """
        Locates or downloads the necessary GGUF model as defined in config.yaml.
        Precedence:
        1. Explicitly provided `model_path`
        2. Configured path/download via config.yaml
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
        if os.path.exists(config_path):
            return config_path
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        filename = model_config.get("filename")
        if not repo_id or not filename:
            raise ValueError(f"Model file not found at {config_path} and auto-download parameters (repo_id, filename) are missing in config.yaml")
            
        print(f"[SLMRag] Model not found at configured path. Auto-downloading...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(config_path)
        )
        if downloaded != config_path and os.path.exists(downloaded):
            os.rename(downloaded, config_path)
        return config_path

    def answer(self, chunks: list, question: str, instruction: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
        """
        Synthesizes an answer based on document chunks, user question, and user instruction.
        """
        # Format the text chunks for context
        formatted_chunks = ""
        for i, chunk in enumerate(chunks):
            formatted_chunks += f"--- Chunk {i+1} ---\n{chunk.strip()}\n\n"
            
        # Build strict ChatML template prompt for Qwen 2.5
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
        
        # Generation configuration for CPU inference
        response = self.llm(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["<|im_end|>", "<|im_start|>", "--- Chunk"] # Prevent hallucinating or spilling over
        )
        
        answer_text = response["choices"][0]["text"].strip()
        return answer_text
