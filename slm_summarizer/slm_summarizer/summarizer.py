import os
import sys
import yaml
import json

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
        os.environ.get("SLM_SUMMARIZER_CONFIG"),
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
                print(f"[SLMSummarizer] Warning: Failed to load config from {path}: {e}")
    return {}

class SLMSummarizer:
    """
    A CPU-optimized text summarization agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    Handles short texts in a single pass, and dynamically applies recursive Map-Reduce chunking
    for larger documents to minimize latency and memory spikes.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=8192, n_threads=4):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )
            
        # Resolve the ONNX model directory
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMSummarizer] Loading ONNX model from: {self.model_path}...")
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
        model_config = config.get("models", {}).get("summarizer")
        if not model_config:
            raise ValueError("models.summarizer configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.summarizer in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        
        # Check if genai_config.json exists in config_path or its subdirectories
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Model directory not found at {config_path} and auto-download parameters (repo_id) are missing in config.yaml")
            
        print(f"[SLMSummarizer] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        # Download ONNX GenAI model weights & configs, ignoring CUDA/DirectML specific files
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        # Scan again to find the actual directory containing genai_config.json (handles nested exports)
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                print(f"[SLMSummarizer] Resolved model directory containing genai_config.json: {root}")
                return root
                
        return config_path

    def _chunk_text(self, text: str, chunk_size: int) -> list[str]:
        """
        Splits text into chunks of roughly `chunk_size` characters, attempting to split
        on paragraph boundaries (\n\n or \n) or word spaces first.
        """
        if not text:
            return []
            
        chunks = []
        words = text.split(" ")
        current_chunk = []
        current_len = 0
        
        for word in words:
            word_len = len(word) + 1
            if current_len + word_len > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_len = word_len
            else:
                current_chunk.append(word)
                current_len += word_len
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def _generate_summary(self, text: str, system_prompt: str, max_tokens: int, temperature: float) -> str:
        """
        Executes text generation using local ONNX Runtime GenAI.
        """
        # Format in ChatML template syntax
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(prompt)
        
        params = og.GeneratorParams(self.model)
        
        # total_max_length is input tokens + new generated tokens
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


    def summarize(self, text: str, format: str = "bullet_points", max_length: int = 256, 
                  instruction: str = "", chunk_size: int = 4000, temperature: float = 0.0) -> str:
        """
        Summarizes the given text.
        
        Parameters:
          text: The raw text document to summarize.
          format: The output format. Options are:
                  - 'bullet_points': A bulleted list of key takeaways.
                  - 'paragraph': A concise summary paragraph.
                  - 'tldr': A single-sentence TL;DR summary.
          max_length: The maximum number of tokens to generate for the summary.
          instruction: An optional focus or style instruction.
          chunk_size: Character threshold for Map-Reduce chunking.
          temperature: Controls generation diversity (default 0.0 for deterministic results).
        """
        text = text.strip()
        if not text:
            return ""

        format_instr = {
            "bullet_points": "Provide a bulleted list of the key takeaways and main points from the text. Use '-' prefix for bullets.",
            "paragraph": "Write a concise, cohesive paragraph summarizing the key points of the text.",
            "tldr": "Write a one-sentence TL;DR summary capturing the most important takeaway of the text."
        }.get(format.lower(), "Provide a concise summary of the text.")

        if instruction:
            format_instr += f" Style/Focus restriction: {instruction}"

        # If text is small enough, do a direct single-pass summary
        if len(text) <= chunk_size:
            system_prompt = (
                "You are an expert summarization assistant. Your goal is to produce a precise, accurate, "
                "and helpful summary of the user's text. Avoid hallucinating details not mentioned in the source.\n"
                f"Instruction: {format_instr}"
            )
            return self._generate_summary(text, system_prompt, max_length, temperature)
            
        # Map-Reduce Flow for larger texts
        print(f"[SLMSummarizer] Text length ({len(text)} chars) exceeds chunk_size ({chunk_size}). Applying Map-Reduce...")
        
        chunks = self._chunk_text(text, chunk_size)
        print(f"[SLMSummarizer] Document split into {len(chunks)} chunks.")
        
        map_prompt = (
            "You are a precise reading assistant. Summarize the key points of the following section "
            "of a larger document. Keep the summary short and highlight key information."
        )
        if instruction:
            map_prompt += f" Focus on: {instruction}"
            
        chunk_summaries = []
        for idx, chunk in enumerate(chunks):
            print(f"[SLMSummarizer] Summarizing chunk {idx + 1}/{len(chunks)}...")
            summary = self._generate_summary(chunk, map_prompt, max_tokens=150, temperature=0.0)
            chunk_summaries.append(summary)
            
        combined_summaries = "\n\n".join(chunk_summaries)
        
        while len(combined_summaries) > chunk_size:
            print(f"[SLMSummarizer] Combined summaries too long ({len(combined_summaries)} chars). Reducing further...")
            sub_chunks = self._chunk_text(combined_summaries, chunk_size)
            sub_summaries = []
            for idx, sub_chunk in enumerate(sub_chunks):
                print(f"[SLMSummarizer] Summarizing sub-chunk {idx + 1}/{len(sub_chunks)}...")
                summary = self._generate_summary(sub_chunk, map_prompt, max_tokens=150, temperature=0.0)
                sub_summaries.append(summary)
            combined_summaries = "\n\n".join(sub_summaries)
            
        print(f"[SLMSummarizer] Generating final summary in target format: {format}...")
        final_system_prompt = (
            "You are an expert summarization assistant. Combine and synthesize the following chunk summaries "
            "into a single final unified summary. Ensure no external facts are introduced.\n"
            f"Instruction: {format_instr}"
        )
        return self._generate_summary(combined_summaries, final_system_prompt, max_length, temperature)

    def summarize_json(self, json_input, format: str = "bullet_points", **kwargs) -> str:
        """
        Accepts a JSON string or dict containing the fields:
          - 'passage' (or 'text'): The text/passage to summarize
          - 'prompt' (or 'instruction'): The focus or style prompt/instruction
          - 'size' (or 'max_length'): The desired target token count (max length) for the summary
          - 'format' (or 'type'): Optional summary format ('bullet_points', 'paragraph', 'tldr')
        
        Returns the generated summary text.
        """
        if isinstance(json_input, str):
            try:
                data = json.loads(json_input)
            except Exception as e:
                raise ValueError(f"Invalid JSON string input: {e}")
        elif isinstance(json_input, dict):
            data = json_input
        else:
            raise TypeError("json_input must be a JSON string or dict")

        passage = data.get("passage") or data.get("text")
        if not passage:
            raise ValueError("JSON input must contain a 'passage' or 'text' key.")

        prompt = data.get("prompt") or data.get("instruction") or ""
        
        size = data.get("size") or data.get("max_length") or 256
        if isinstance(size, str):
            try:
                size = int(size)
            except ValueError:
                size = 256
                
        target_format = data.get("format") or data.get("type") or format
        
        return self.summarize(
            text=passage,
            format=target_format,
            max_length=size,
            instruction=prompt,
            **kwargs
        )
