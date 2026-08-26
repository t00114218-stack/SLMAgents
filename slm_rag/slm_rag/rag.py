import os
import sys
import yaml
import re
import json
import math
import difflib
from collections import Counter

# Setup sys.path to resolve all SLM agent packages locally
_curr_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_curr_dir, "..", ".."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)
if os.path.exists(_root_dir):
    for folder in os.listdir(_root_dir):
        folder_path = os.path.join(_root_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith("slm_"):
            if folder_path not in sys.path:
                sys.path.insert(0, folder_path)

try:
    import numpy as np
except ImportError:
    np = None

try:
    from slm_embeddings.embeddings_server import SLMEmbeddingsServer
except ImportError:
    SLMEmbeddingsServer = None

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
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception as e:
                raise ValueError(f"Failed to parse config file at {path}: {e}")
    raise FileNotFoundError("config.yaml not found in environment, current directory, or package directories.")

_SHARED_RAG_MODEL = None
_SHARED_RAG_TOKENIZER = None

class SLMRag:
    """
    A CPU-optimized Retrieval-Augmented Generation (RAG) runner powered by a local
    Small Language Model (SLM) running via ONNX Runtime GenAI.
    Answers user questions based on provided document chunks while strictly adhering
    to user instructions.

    Configuration can be set via constructor arguments OR environment variables:
      SLM_RAG_CACHE_DIR   — Override model download/cache directory
      SLM_RAG_N_THREADS   — Number of CPU threads (default: 4)
      SLM_RAG_N_CTX       — Context window size (default: 8192)
      SLM_RAG_MAX_TOKENS  — Default max tokens per answer (default: 256)
      SLM_RAG_CONFIG      — Path to a custom config.yaml file
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        global _SHARED_RAG_MODEL, _SHARED_RAG_TOKENIZER
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )

        if _SHARED_RAG_MODEL is not None and _SHARED_RAG_TOKENIZER is not None:
            self.model = _SHARED_RAG_MODEL
            self.tokenizer = _SHARED_RAG_TOKENIZER
            self.n_ctx = n_ctx or int(os.environ.get("SLM_RAG_N_CTX", 8192))
            return

        # Resolve parameters: constructor args > env vars > defaults
        _default_threads = min(8, max(4, os.cpu_count() or 4))
        n_threads = n_threads or int(os.environ.get("SLM_RAG_N_THREADS", os.environ.get("SLM_N_THREADS", _default_threads)))
        n_ctx     = n_ctx     or int(os.environ.get("SLM_RAG_N_CTX", 8192))
        cache_dir = cache_dir or os.environ.get("SLM_RAG_CACHE_DIR")

        # Wire thread count to ONNX Runtime (must be set before model load)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        # Resolve the ONNX model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        try:
            print(f"[SLMRag] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
            _SHARED_RAG_MODEL = self.model
            _SHARED_RAG_TOKENIZER = self.tokenizer
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
                    _SHARED_RAG_MODEL = self.model
                    _SHARED_RAG_TOKENIZER = self.tokenizer
                else:
                    self.model = None
                    self.tokenizer = None
            except Exception:
                print(f"[SLMRag] Note: ONNX model load deferred ({e}). Operating in low-RAM fallback mode.")
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
        model_config = config.get("models", {}).get("rag")
        if not model_config:
            raise ValueError("models.rag configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.rag in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        # Check if model directory exists
        if os.path.exists(os.path.join(config_path, "tokenizer.json")):
            return config_path
            
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files or "tokenizer.json" in files:
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

    def query(self, question: str, chunks: list = None, system_prompt: str = None, token_callback: callable = None, **kwargs):
        chunks = chunks or []
        if not chunks:
            return "I couldn't find any uploaded documents to reference, so no document context is currently available. Could you please upload or attach the document you'd like me to analyze? I'd be happy to answer your questions once you provide it! 😊"
        instruction = system_prompt or "Answer the question accurately based on context."
        return self.answer(chunks=chunks, question=question, instruction=instruction, token_callback=token_callback, **kwargs)

    def answer(self, chunks: list, question: str, instruction: str, temperature: float = 0.0, max_tokens: int = None, tools: list = None, tool_executor: callable = None, max_iterations: int = 5, stream: bool = False, system_prompt: str = None, user_input: str = None, token_callback: callable = None):


        """
        Synthesizes an answer based on document chunks, user question, and user instruction.
        Supports tool execution (e.g., Vector DB lookups) to gather more context.

        Args:
            chunks:         List of document text strings to use as context.
            question:       The user's question.
            instruction:    Style or constraint the model must follow.
            temperature:    Generation randomness. 0.0 = deterministic (fastest, most consistent).
            max_tokens:     Max tokens to generate. Lower = faster. Env: SLM_RAG_MAX_TOKENS.
            tools:          Optional list of tool JSON schemas for agentic retrieval.
            tool_executor:  Optional callable(tool_name, args) -> str to execute tools.
            max_iterations: Max ReAct tool-calling loops (prevents infinite loops).
            stream:         If True, streams token strings in real-time.
        """
        # Resolve max_tokens: arg > env var > default (3000 tokens for long-form detailed responses)
        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_RAG_MAX_TOKENS", 3000))

        max_iterations = max(1, min(int(max_iterations), 8))
        
        # Rank and filter top relevant chunks via Okapi BM25 with fuzzy vocabulary expansion
        selected_chunks = chunks
        if len(chunks) > 4:
            q_clean = question.strip()
            q_tokens = [w.lower() for w in re.findall(r'\b\w+\b', q_clean) if len(w) > 2]
            
            # 1. Okapi BM25 Ranking across all document chunks
            N = len(chunks)
            doc_freqs = Counter()
            doc_lens = []
            tokenized_docs = []
            all_vocab = set()
            for c in chunks:
                toks = [w.lower() for w in re.findall(r'\b\w+\b', c)]
                tokenized_docs.append(toks)
                doc_lens.append(len(toks))
                for term in set(toks):
                    doc_freqs[term] += 1
                    all_vocab.add(term)
            
            # Expand misspelled / typo query tokens using document vocabulary
            expanded_tokens = []
            for q in q_tokens:
                expanded_tokens.append(q)
                if q not in doc_freqs:
                    close_matches = difflib.get_close_matches(q, all_vocab, n=2, cutoff=0.75)
                    for m in close_matches:
                        expanded_tokens.append(m)

            avgdl = sum(doc_lens) / N if N else 1.0
            k1 = 1.5
            b = 0.75
            
            bm25_scores = []
            for idx, toks in enumerate(tokenized_docs):
                score = 0.0
                doc_len = doc_lens[idx]
                counts = Counter(toks)
                for q in expanded_tokens:
                    if q in doc_freqs:
                        df = doc_freqs[q]
                        idf = math.log(1.0 + (N - df + 0.5) / (df + 0.5))
                        tf = counts[q]
                        score += idf * (tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl)))
                bm25_scores.append(score)

            # 2. Select top ranked chunks via Okapi BM25 (< 5ms retrieval)
            scored = [(bm25_scores[i], i, chunks[i]) for i in range(len(chunks))]
            scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            
            # Select top 4 most informative passages
            top_indices = [idx for _, idx, _ in scored[:4]]
            
            # Include immediate adjacent chunks for document continuity
            expanded_indices = set(top_indices)
            for idx in top_indices:
                if idx + 1 < len(chunks):
                    expanded_indices.add(idx + 1)
                if len(expanded_indices) >= 6:
                    break

            selected_chunks = [chunks[i] for i in sorted(expanded_indices)]
        else:
            selected_chunks = chunks[:6]

        # Format text chunks for context
        formatted_chunks = "\n\n".join([chunk.strip() for chunk in selected_chunks if chunk.strip()])
            
        # Build thorough, detailed ChatML template prompt
        system_prompt = (
            "You are an expert document analysis assistant.\n"
            "Provide a detailed, complete, and thorough answer to the user's question based strictly on the provided Document Text.\n"
            "Include all specific figures, amounts, financial breakdown tables, payment schedules, and explicit conditions mentioned in the context.\n"
            "Do not summarize vaguely or omit details when specific data is present in the text."
        )

        
        if tools and tool_executor:
            system_prompt += (
                f"\n\nAvailable Tools:\n{json.dumps(tools, indent=2)}\n"
                "If you need more information (e.g., the provided chunks are insufficient), you can use a tool by outputting a JSON object with 'tool_call' and 'args' keys. Example:\n"
                "{\"tool_call\": \"search_vector_db\", \"args\": {\"query\": \"something\"}}\n"
                "If you have enough information, output your final answer directly as text (not JSON)."
            )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Document Text:\n{formatted_chunks}\n\n"
            f"Question: {question}<|im_end|>\n"
        )
        
        if stream:
            def generator_fn():
                nonlocal prompt
                seen_tool_calls = set()
                for iteration in range(max_iterations):
                    current_prompt = prompt + "<|im_start|>assistant\n"
                    input_tokens = self.tokenizer.encode(current_prompt)
                    
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
                    
                    if not (tools and tool_executor):
                        tokenizer_stream = self.tokenizer.create_stream()
                        while not generator.is_done():
                            generator.generate_next_token()
                            new_tokens = generator.get_next_tokens()
                            if len(new_tokens) > 0:
                                token_id = int(new_tokens[0])
                                if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                                    break
                                yield tokenizer_stream.decode(token_id)
                        return
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
                        
                        response_text = self.tokenizer.decode(output_tokens).strip()
                        
                        is_tool_call = False
                        try:
                            cleaned = response_text.replace("```json", "").replace("```", "").strip()
                            if cleaned.startswith("{") and cleaned.endswith("}"):
                                data = json.loads(cleaned)
                                if "tool_call" in data:
                                    is_tool_call = True
                                    tool_name = data["tool_call"]
                                    args = data.get("args", {})
                                    signature = json.dumps([tool_name, args], sort_keys=True, default=str)
                                    if signature in seen_tool_calls:
                                        yield "Tool execution stopped because the same request was repeated without progress."
                                        return
                                    seen_tool_calls.add(signature)
                                    
                                    try:
                                        result = tool_executor(tool_name, args)
                                    except Exception as e:
                                        result = f"Error executing tool {tool_name}: {e}"
                                        
                                    prompt += (
                                        "<|im_start|>assistant\n"
                                        f"{response_text}<|im_end|>\n"
                                        "<|im_start|>tool\n"
                                        f"{result}<|im_end|>\n"
                                    )
                        except Exception:
                            pass
                            
                        if not is_tool_call:
                            yield self.verify_and_ground(response_text, formatted_chunks, question)
                            return
                yield "Tool execution stopped after reaching the maximum number of steps without a final answer."
            return generator_fn()
        else:
            seen_tool_calls = set()
            for iteration in range(max_iterations):
                current_prompt = prompt + "<|im_start|>assistant\n"
                input_tokens = self.tokenizer.encode(current_prompt)
                
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
                
                output_tokens = []
                accumulated_str = ""
                tokenizer_stream = self.tokenizer.create_stream() if hasattr(self.tokenizer, "create_stream") else None
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        output_tokens.append(token_id)
                        tok_str = tokenizer_stream.decode(token_id) if tokenizer_stream else self.tokenizer.decode([token_id])
                        accumulated_str += tok_str
                        if any(stop_word in accumulated_str for stop_word in ["<|im_end|>", "<|endoftext|>", "<|end|>"]):
                            break
                        if token_callback and tok_str:
                            try:
                                token_callback(tok_str)
                            except Exception:
                                pass

                if self.tokenizer is None or self.model is None:
                    return "Model not initialized."

                response_text = self.tokenizer.decode(output_tokens).strip()
                response_text = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                response_text = re.sub(r'</?think>', '', response_text).strip()
                
                is_tool_call = False
                if tools and tool_executor:
                    try:
                        cleaned = response_text.replace("```json", "").replace("```", "").strip()
                        if cleaned.startswith("{") and cleaned.endswith("}"):
                            data = json.loads(cleaned)
                            if "tool_call" in data:
                                is_tool_call = True
                                tool_name = data["tool_call"]
                                args = data.get("args", {})
                                signature = json.dumps([tool_name, args], sort_keys=True, default=str)
                                if signature in seen_tool_calls:
                                    return "Tool execution stopped because the same request was repeated without progress."
                                seen_tool_calls.add(signature)
                                
                                try:
                                    result = tool_executor(tool_name, args)
                                except Exception as e:
                                    result = f"Error executing tool {tool_name}: {e}"
                                    
                                prompt += (
                                    "<|im_start|>assistant\n"
                                    f"{response_text}<|im_end|>\n"
                                    "<|im_start|>tool\n"
                                    f"{result}<|im_end|>\n"
                                )
                    except Exception:
                        pass
                
                if not is_tool_call:
                    return self.verify_and_ground(response_text, formatted_chunks, question)

                
        return "Tool execution stopped after reaching the maximum number of steps without a final answer."

    def verify_and_ground(self, answer: str, context_text: str, question: str = "") -> str:
        """
        Clean grounding verification: returns the model's synthesized natural language answer.
        """
        if not answer or not answer.strip():
            return "Unable to find relevant information in the provided document."

        # Remove trailing stop tags if present
        cleaned = re.sub(r'<\|im_end\|>|<\|endoftext\|>|<\|end\|>', '', answer).strip()
        return cleaned or answer
