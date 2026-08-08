import os
import sys
from llama_cpp import Llama

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
        Locates or downloads the necessary Qwen 2.5 1.5B GGUF model.
        Precedence:
        1. Explicitly provided `model_path`
        2. Current directory model file (`qwen2.5-1.5b-instruct-q4_k_m.gguf`)
        3. User cache directory (`~/.cache/slm_rag/`)
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)
            
        # 1. Check current working directory
        cwd_model = os.path.join(os.getcwd(), "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if os.path.exists(cwd_model):
            return cwd_model
            
        # 2. Check user cache directory (~/.cache/slm_rag/)
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/slm_rag")
        os.makedirs(cache_dir, exist_ok=True)
        
        cached_model = os.path.join(cache_dir, "qwen2.5-1.5b-instruct-q4_k_m.gguf")
        if not os.path.exists(cached_model):
            print(f"[SLMRag] Model not found locally. Auto-downloading to cache: {cached_model}...")
            from huggingface_hub import hf_hub_download
            
            # Download Qwen 2.5 1.5B Instruct model
            hf_hub_download(
                repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
                filename="qwen2.5-1.5b-instruct-q4_k_m.gguf",
                local_dir=cache_dir
            )
            
        return cached_model

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
