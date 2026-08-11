import os
import sys
import yaml
import json
import re

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

try:
    from duckduckgo_search import DDGS
except ImportError:
    DDGS = None

class SLMSearchOrchestrator:
    """
    A local CPU-optimized Search query planner and retriever powered by a local MIT-licensed Phi-3.5 model
    running via ONNX Runtime GenAI. Integrates with DuckDuckGo to aggregate and structure web snippets.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_SEARCH_ORCHESTRATOR_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_SEARCH_ORCHESTRATOR_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_SEARCH_ORCHESTRATOR_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMSearchOrchestrator] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        # Config loading helper
        config_paths = [
            "./config.yaml",
            "../config.yaml",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        ]
        config = {}
        config_file_path = ""
        for path in config_paths:
            if path and os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        config, config_file_path = yaml.safe_load(f) or {}, os.path.abspath(path)
                        break
                except Exception:
                    pass

        model_config = config.get("models", {}).get("search_orchestrator", {})
        config_path = model_config.get("path", "../../models/phi-3.5-mini-instruct-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "microsoft/Phi-3.5-mini-instruct-onnx")
        print(f"[SLMSearchOrchestrator] ONNX Model not found at configured path. Auto-downloading...")
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

    def generate_queries(self, user_query: str) -> list[str]:
        """Expands the user query into up to 3 distinct search terms."""
        system_prompt = (
            "You are an expert search planner.\n"
            "Analyze the user query and generate exactly 3 optimized search string variations.\n"
            "Output your search queries inside a single ```json ... ``` list code block, matching the format:\n"
            "[\n"
            "  \"search query 1\",\n"
            "  \"search query 2\",\n"
            "  \"search query 3\"\n"
            "]\n"
            "Do not output explanations outside the code block."
        )

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Query: {user_query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_tokens = self.tokenizer.encode(full_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + 256, temperature=0.0)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        
        response_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                response_text += self.tokenizer.decode(new_tokens)

        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = match.group(1).strip() if match else "[]"
        try:
            queries = json.loads(json_str)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)]
        except Exception:
            pass
        return [user_query]

    def execute_search(self, query: str, max_results: int = 3) -> list[dict]:
        """Queries DuckDuckGo search for snippets."""
        mock_results = [
            {
                "title": f"Mock result for: {query}",
                "href": f"https://example.com/mock-{hash(query) % 100}",
                "body": f"Mock details describing relevant search results for query: {query}"
            }
        ]
        
        if DDGS is None:
            return mock_results
            
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                if not results:
                    return mock_results
                return [
                    {
                        "title": r.get("title", ""),
                        "href": r.get("href", ""),
                        "body": r.get("body", "")
                    }
                    for r in results
                ]
        except Exception as e:
            print(f"Warning: DuckDuckGo query failed: {e}")
            return mock_results

    def retrieve(self, user_query: str, max_results_per_query: int = 2) -> list[dict]:
        """Performs full search aggregation workflow."""
        queries = self.generate_queries(user_query)
        all_results = []
        seen_links = set()
        
        for q in queries:
            results = self.execute_search(q, max_results=max_results_per_query)
            for r in results:
                link = r.get("href")
                if link and link not in seen_links:
                    seen_links.add(link)
                    all_results.append(r)
                    
        return all_results
