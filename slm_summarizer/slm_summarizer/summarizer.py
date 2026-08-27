import os
import sys
import re
import yaml
import json

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
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception as e:
                print(f"[SLMSummarizer] Warning: Failed to load config from {path}: {e}")
    return {}, ""

class SLMSummarizer:
    """
    A CPU-optimized text summarization agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    Handles short texts in a single pass, and dynamically applies recursive Map-Reduce chunking
    for larger documents to minimize latency and memory spikes.

    Configuration can be set via constructor arguments OR environment variables:
      SLM_SUMMARIZER_CACHE_DIR            — Override model download/cache directory
      SLM_SUMMARIZER_N_THREADS            — Number of CPU threads (default: 4)
      SLM_SUMMARIZER_N_CTX               — Context window size (default: 8192)
      SLM_SUMMARIZER_MAX_LENGTH          — Default max output tokens (default: 256)
      SLM_SUMMARIZER_MAX_CORRECTION_LOOPS — Default evaluator loop count (default: 0)
      SLM_SUMMARIZER_CONFIG              — Path to a custom config.yaml file
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )

        # Resolve parameters: constructor args > env vars > defaults
        n_threads = n_threads or int(os.environ.get("SLM_SUMMARIZER_N_THREADS", os.environ.get("SLM_N_THREADS", 2)))
        n_ctx     = n_ctx     or int(os.environ.get("SLM_SUMMARIZER_N_CTX", 8192))
        cache_dir = cache_dir or os.environ.get("SLM_SUMMARIZER_CACHE_DIR")

        # Wire thread count to ONNX Runtime (must be set before model load)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        # Resolve the ONNX model directory
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        try:
            print(f"[SLMSummarizer] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as e:
            try:
                main_mod = sys.modules.get("main") or sys.modules.get("__main__")
                if not main_mod or not hasattr(main_mod, "get_shared_onnx_genai"):
                    try:
                        import importlib
                        main_mod = importlib.import_module("main")
                    except ImportError:
                        main_mod = None
                if main_mod and hasattr(main_mod, "get_shared_onnx_genai"):
                    self.model, self.tokenizer = main_mod.get_shared_onnx_genai()
                else:
                    self.model = None
                    self.tokenizer = None
            except Exception:
                print(f"[SLMSummarizer] Note: ONNX model load deferred ({e}). Operating in low-RAM fallback mode.")
                self.model = None
                self.tokenizer = None

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        """
        Locates or downloads the necessary ONNX model as defined in config.yaml.
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        # Check config.yaml
        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("summarizer")
        if not model_config:
            raise ValueError("models.summarizer configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.summarizer in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        # Check if genai_config.json or model files exist in config_path
        if os.path.exists(config_path):
            for root, dirs, files in os.walk(config_path):
                if any(f in files for f in ["genai_config.json", "model_q4.onnx", "model.onnx"]):
                    return root
            return config_path

            
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

    def _generate_summary(self, text: str, system_prompt: str, max_tokens: int, temperature: float, stream: bool = False, token_callback: callable = None):
        """
        Executes text generation using local ONNX Runtime GenAI.
        """
        if not system_prompt:
            system_prompt = "You are a concise, structured AI text summarizer. Write a clear summary highlighting the key points directly with bullet points. Do not output <think> tags."
        else:
            system_prompt += " Do not output <think> tags."

        if not self.model or not self.tokenizer:
            paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
            if format == "bullet_points":
                return "\n".join([f"- {p}" for p in paragraphs[:3]]) if paragraphs else "- Summary unavailable."
            elif format == "tldr":
                return f"TL;DR: {paragraphs[0]}" if paragraphs else "TL;DR: Summary unavailable."
            else:
                return " ".join(paragraphs[:2]) if paragraphs else text[:300]

        # Format in ChatML template syntax
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}\nThink step-by-step inside <think>...</think> tags before finalizing your summary.<|im_end|>\n"
            "<|im_start|>user\n"
            f"{text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(prompt)
        
        params = og.GeneratorParams(self.model)
        total_max_length = len(input_tokens) + max_tokens
        
        search_options = {
            "max_length": total_max_length,
            "temperature": temperature,
            "repetition_penalty": 1.15
        }
        params.set_search_options(**search_options)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        
        if stream:
            def token_generator():
                tokenizer_stream = self.tokenizer.create_stream()
                in_think = False
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        decoded_chunk = tokenizer_stream.decode(token_id)
                        if "<think>" in decoded_chunk:
                            in_think = True
                            continue
                        if "</think>" in decoded_chunk:
                            in_think = False
                            continue
                        if not in_think:
                            yield decoded_chunk
            return token_generator()
        else:
            output_tokens = []
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    token_id = int(new_tokens[0])
                    if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                        break
                    output_tokens.append(token_id)
                    if token_callback:
                        try:
                            tok_str = self.tokenizer.decode([token_id])
                            if tok_str and "<think>" not in tok_str and "</think>" not in tok_str:
                                token_callback(tok_str)
                        except Exception:
                            pass
                
            raw_text = self.tokenizer.decode(output_tokens).strip()
            if "<think>" in raw_text:
                if "</think>" in raw_text:
                    post_think = raw_text.split("</think>", 1)[1].strip()
                    raw_text = post_think if post_think else raw_text.replace("<think>", "").replace("</think>", "").strip()
                else:
                    raw_text = raw_text.replace("<think>", "").strip()
            cleaned_text = re.sub(r"<\|im_\w+\|>", "", raw_text).strip()
            return cleaned_text


    def _evaluate_summary(self, original_text: str, summary: str, instruction: str, max_tokens: int = 128, temperature: float = 0.0) -> dict:
        """
        Evaluates a generated summary against the original text and instructions.
        Returns a dictionary with 'critique' and 'needs_correction'.
        """
        system_prompt = (
            "You are an expert summary evaluator. Your task is to critique a summary against its original text. "
            "Check for hallucinations, missing critical points, and adherence to the instruction. "
            "Output your evaluation strictly as JSON with keys 'critique' (string) and 'needs_correction' (boolean)."
        )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Original Text:\n{original_text}\n\n"
            f"Instruction provided for summary: {instruction}\n\n"
            f"Summary to evaluate:\n{summary}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(prompt)
        params = og.GeneratorParams(self.model)
        
        search_options = {
            "max_length": len(input_tokens) + max_tokens,
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
                
        response_text = self.tokenizer.decode(output_tokens).strip()
        
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            return {
                "critique": data.get("critique", "No critique provided."),
                "needs_correction": data.get("needs_correction", False)
            }
        except Exception as e:
            print(f"[SLMSummarizer] Warning: Failed to parse evaluator JSON: {e}")
            return {"critique": "Failed to parse", "needs_correction": False}

    def _correct_summary(self, original_text: str, summary: str, critique: str, instruction: str, max_tokens: int, temperature: float = 0.0) -> str:
        """
        Corrects a summary based on a critique.
        """
        system_prompt = (
            "You are an expert summary corrector. Your task is to rewrite a summary to fix the issues "
            "identified in the critique, ensuring it matches the original text and adheres to the instruction. "
            "Output only the corrected summary, without any conversational padding."
        )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Original Text:\n{original_text}\n\n"
            f"Instruction for summary: {instruction}\n\n"
            f"Current Summary:\n{summary}\n\n"
            f"Critique:\n{critique}\n\n"
            "Please rewrite the summary based on the critique.<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(prompt)
        params = og.GeneratorParams(self.model)
        
        search_options = {
            "max_length": len(input_tokens) + max_tokens,
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

    def summarize(self, text: str, format: str = "bullet_points", max_length: int = None, 
                  instruction: str = "", chunk_size: int = 4000, temperature: float = 0.7,
                  max_correction_loops: int = None, stream: bool = False, system_prompt: str = None, user_input: str = None, token_callback: callable = None):
        """
        Summarizes the given text.

        Args:
            text:                 The raw text document to summarize.
            format:               Output format: 'bullet_points', 'paragraph', or 'tldr'.
            max_length:           Max tokens for the output. Lower = faster.
                                  Env: SLM_SUMMARIZER_MAX_LENGTH (default: 256).
            instruction:          Optional focus or style constraint (e.g. 'Write like a journalist').
            chunk_size:           Character threshold for Map-Reduce chunking (default: 4000).
            temperature:          0.0 = deterministic, 0.7+ = creative (default: 0.0).
            max_correction_loops: Evaluator-corrector iterations. 0 = disabled (fastest).
                                  Env: SLM_SUMMARIZER_MAX_CORRECTION_LOOPS (default: 0).
            stream:               If True, streams token strings in real-time.
        """
        # Resolve defaults: arg > env var > hardcoded default
        if max_length is None:
            max_length = int(os.environ.get("SLM_SUMMARIZER_MAX_LENGTH", 3000))
        if max_correction_loops is None:
            max_correction_loops = int(os.environ.get("SLM_SUMMARIZER_MAX_CORRECTION_LOOPS", 0))
        text = text.strip()
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not text:
            return ""

        format_instr = {
            "bullet_points": "Provide a bulleted list of the key takeaways and main points from the text. Use '-' prefix for bullets.",
            "paragraph": "Write a concise, cohesive paragraph summarizing the key points of the text.",
            "tldr": "Write a one-sentence TL;DR summary capturing the most important takeaway of the text."
        }.get(format.lower(), "Provide a concise summary of the text.")

        if instruction:
            format_instr += f" Style/Focus restriction: {instruction}"

        # 1. Generate Initial Summary
        initial_summary = ""
        # If text is small enough, do a direct single-pass summary
        if len(text) <= chunk_size:
            if stream:
                system_prompt = (
                    "You are an articulate, expert AI summarization assistant. Produce a clear, comprehensive, "
                    "and beautifully formatted summary of the user's text. Avoid hallucinating details.\n"
                    f"Instruction: {format_instr}"
                )
            else:
                system_prompt = (
                    "You are an articulate, expert AI summarization assistant. Produce a clear, comprehensive, "
                    "and beautifully formatted summary of the user's text. Avoid hallucinating details.\n"
                    f"Instruction: {format_instr}"
                )
            initial_summary = self._generate_summary(text, system_prompt, max_length, temperature, stream=stream)
            if stream:
                return initial_summary
        else:
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
            
            max_reduce_rounds = 4
            for _ in range(max_reduce_rounds):
                if len(combined_summaries) <= chunk_size:
                    break
                previous_length = len(combined_summaries)
                print(f"[SLMSummarizer] Combined summaries too long ({len(combined_summaries)} chars). Reducing further...")
                sub_chunks = self._chunk_text(combined_summaries, chunk_size)
                sub_summaries = []
                for idx, sub_chunk in enumerate(sub_chunks):
                    print(f"[SLMSummarizer] Summarizing sub-chunk {idx + 1}/{len(sub_chunks)}...")
                    summary = self._generate_summary(sub_chunk, map_prompt, max_tokens=150, temperature=0.0)
                    sub_summaries.append(summary)
                combined_summaries = "\n\n".join(sub_summaries)
                if len(combined_summaries) >= previous_length:
                    combined_summaries = combined_summaries[:chunk_size]
                    break
                
            print(f"[SLMSummarizer] Generating final initial summary in target format: {format}...")
            if stream:
                final_system_prompt = (
                    "You are an expert summarization assistant. Combine and synthesize the following chunk summaries "
                    "into a single final unified summary. Ensure no external facts are introduced.\n"
                    "You MUST first think step-by-step about the request and summarize key aspects inside <thought>...</thought> tags, "
                    "and then provide your final summary.\n"
                    f"Instruction: {format_instr}"
                )
            else:
                final_system_prompt = (
                    "You are an expert summarization assistant. Combine and synthesize the following chunk summaries "
                    "into a single final unified summary. Ensure no external facts are introduced.\n"
                    f"Instruction: {format_instr}"
                )
            initial_summary = self._generate_summary(combined_summaries, final_system_prompt, max_length, temperature, stream=stream)
            if stream:
                return initial_summary

        # 2. Evaluator/Corrector Loop
        current_summary = initial_summary
        
        for iteration in range(max_correction_loops):
            print(f"[SLMSummarizer] Evaluator iteration {iteration + 1}/{max_correction_loops}...")
            # We evaluate against the original text if it fits, else the combined_summaries is safer.
            eval_text = text if len(text) <= chunk_size else combined_summaries
            
            evaluation = self._evaluate_summary(eval_text, current_summary, format_instr, temperature=temperature)
            
            if not evaluation.get("needs_correction"):
                print("[SLMSummarizer] Evaluator accepted the summary.")
                break
                
            print(f"[SLMSummarizer] Summary needs correction. Critique: {evaluation.get('critique')}")
            current_summary = self._correct_summary(
                eval_text, 
                current_summary, 
                evaluation.get('critique', ''), 
                format_instr, 
                max_length, 
                temperature=temperature
            )
            
        return current_summary

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
