import os
import sys
import yaml
import re
import json

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
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )

        # Resolve parameters: constructor args > env vars > defaults
        n_threads = n_threads or int(os.environ.get("SLM_RAG_N_THREADS", os.environ.get("SLM_N_THREADS", 2)))
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

    def query(self, question: str, chunks: list = None, system_prompt: str = None, **kwargs):
        chunks = chunks or []
        if not chunks:
            return "I couldn't find any uploaded documents to reference, so no document context is currently available. Could you please upload or attach the document you'd like me to analyze? I'd be happy to answer your questions once you provide it! 😊"
        instruction = system_prompt or "Answer the question accurately based on context."
        return self.answer(chunks=chunks, question=question, instruction=instruction, **kwargs)

    def answer(self, chunks: list, question: str, instruction: str, temperature: float = 0.7, max_tokens: int = None, tools: list = None, tool_executor: callable = None, max_iterations: int = 5, stream: bool = False, system_prompt: str = None, user_input: str = None):
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
        import json, re, numpy as np
        # Resolve max_tokens: arg > env var > default
        if max_tokens is None:
            max_tokens = int(os.environ.get("SLM_RAG_MAX_TOKENS", 1024))
        max_iterations = max(1, min(int(max_iterations), 8))
        
        # Dynamic synonym dictionary for domain queries
        SYNONYM_MAP = {
            "retention": ["retention", "retension", "bonus", "joining bonus", "annexure", "compensation", "inr", "rs"],
            "retension": ["retention", "retension", "bonus", "joining bonus", "annexure", "compensation", "inr", "rs"],
            "bonus": ["bonus", "retention", "joining bonus", "incentive", "variable pay", "annexure", "inr", "rs"],
            "salary": ["salary", "compensation", "remuneration", "ctc", "pay", "cost to company", "package", "annexure", "base pay", "fixed pay", "gross", "inr", "rs", "lpa", "lakhs", "allowance", "bonus", "retention"],
            "package": ["package", "ctc", "compensation", "salary", "remuneration", "annexure", "lpa", "lakhs", "inr", "rs", "bonus"],
            "pay": ["pay", "salary", "compensation", "remuneration", "ctc", "wages", "fee"],
            "shares": ["shares", "equity", "esop", "stock", "options", "grant", "allotment", "units", "rsu"],
            "equity": ["equity", "shares", "esop", "stock", "options", "grant", "allotment"],
            "notice": ["notice period", "notice", "resignation", "termination", "severance"],
            "termination": ["termination", "notice period", "severance", "cause", "discharge"]
        }

        # Rank and filter top relevant chunks if document has many chunks
        selected_chunks = chunks
        if len(chunks) > 4:
            q_lower = question.lower()
            # Normalize common typos
            q_lower_norm = q_lower.replace("retension", "retention").replace("salry", "salary").replace("packge", "package")
            q_words = set(re.findall(r'\w+', q_lower_norm))
            stopwords = {"what", "is", "the", "a", "an", "and", "or", "in", "of", "to", "for", "with", "on", "at", "by", "from", "this", "that", "these", "those", "explain", "tell", "me", "about", "here"}
            keywords = [w for w in q_words if w not in stopwords and len(w) > 2]
            
            # Expand keywords with synonyms
            expanded_keywords = set(keywords)
            for kw in keywords:
                if kw in SYNONYM_MAP:
                    expanded_keywords.update(SYNONYM_MAP[kw])
            
            # Try vector embeddings scoring if available
            embed_scores = {}
            try:
                from slm_embeddings.embeddings_server import SLMEmbeddingsServer
                embed_server = SLMEmbeddingsServer()
                q_vec = embed_server.embed(question)
                c_vecs = embed_server.embed(chunks)
                if q_vec and c_vecs:
                    q_arr = np.array(q_vec[0])
                    for i, cv in enumerate(c_vecs):
                        c_arr = np.array(cv)
                        sim = float(np.dot(q_arr, c_arr) / (np.linalg.norm(q_arr) * np.linalg.norm(c_arr) + 1e-12))
                        embed_scores[i] = sim
            except Exception:
                pass

            scored = []
            for i, c in enumerate(chunks):
                c_lower = c.lower()
                # Keyword count + prefix match for minor variations
                score = sum(c_lower.count(kw) * (4 if len(kw) > 5 else 2) for kw in expanded_keywords)
                
                # Check 5-char prefix matches for fuzzy resilience
                for kw in expanded_keywords:
                    if len(kw) >= 5 and kw[:5] in c_lower:
                        score += 3

                if q_lower_norm in c_lower:
                    score += 20
                
                # Incorporate vector similarity score if available
                if i in embed_scores:
                    score += embed_scores[i] * 50

                # Deduct score for Table of Contents pages
                if "table of contents" in c_lower:
                    score -= 50

                # Boost chunks containing monetary / table / Annexure / bonus indicators
                if any(k in q_lower_norm for k in ["salary", "pay", "compensation", "ctc", "package", "remuneration", "money", "amount", "bonus", "retention"]):
                    if any(ind in c_lower for ind in ["annexure 1", "annexure 2", "base in inr", "fixed compensation", "variable pay", "committed pay", "retention", "joining bonus"]):
                        score += 100
                    elif any(ind in c_lower for ind in ["annexure", "ctc", "base", "inr", "rs", "fixed compensation", "variable pay", "bonus"]):
                        score += 40
                    if re.search(r'\b\d{1,3}(,\d{3})+\b', c) or re.search(r'\b\d{5,7}\b', c):
                        score += 60
                elif any(ind in c_lower for ind in ["annexure", "ctc", "inr", "rs.", "lpa", "lakhs", "per annum", "table", "breakup"]):
                    score += 20
                
                scored.append((score, i, c))

            scored.sort(key=lambda x: (x[0], -x[1]), reverse=True)
            # Take top 25 chunks ranked by vector similarity & keyword score to provide comprehensive context
            top_k = min(25, len(scored))
            top_scored = scored[:top_k]
            # Order selected chunks by original document position for coherent context flow
            selected_chunks = [c for _, _, c in sorted(top_scored, key=lambda x: x[1])]
        else:
            selected_chunks = chunks

        # Format up to 25 text chunks for context
        formatted_chunks = "\n\n".join([chunk.strip() for chunk in selected_chunks if chunk.strip()])
            
        # Build strict ChatML template prompt
        system_prompt = (
            "You are a precise, direct RAG QA assistant. Answer the user's question directly using the exact figures, currency amounts (INR / CTC / Base Salary), dates, and facts in the provided text.\n"
            "State the answer directly without listing section headers or meta-commentary."
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
                            yield response_text
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
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        output_tokens.append(token_id)
                        
                if self.tokenizer is None or self.model is None:
                    # Low-RAM fallback keyword search answer
                    q_lower = question.lower()
                    words = [w for w in re.findall(r'\w+', q_lower) if len(w) > 3]
                    matched = [c for c in chunks if any(w in c.lower() for w in words)]
                    if matched:
                        return f"Extracted RAG context: {matched[0]}"
                    return "I don't know."

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
        Clean grounding verification: returns the model's synthesized natural language answer,
        removing any raw placeholder overrides or broken snippet dumps.
        """
        if not answer or not answer.strip():
            # Fallback to relevant document lines if model returned empty output
            q_keywords = [w.lower() for w in re.findall(r'\w+', (question or "").lower()) if len(w) > 3]
            lines = [line.strip() for line in context_text.split("\n") if line.strip()]
            relevant = [l for l in lines if any(kw in l.lower() for kw in (q_keywords or ["retention", "bonus", "salary", "pay", "annexure"]))]
            if relevant:
                return "\n".join(dict.fromkeys(relevant[:6]))
            return context_text[:500]

        # Clean up any leftover placeholder strings or generic templates
        cleaned_ans = re.sub(r'\[Insert Year\]', '', answer, flags=re.IGNORECASE).strip()
        return cleaned_ans or answer
