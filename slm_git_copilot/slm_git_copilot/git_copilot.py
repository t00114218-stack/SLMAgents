import os
import sys
import yaml

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_GIT_COPILOT_CONFIG"),
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
            except Exception:
                pass
    return {}, ""

class SLMGitCopilot:
    """
    A local CPU-optimized Conventional Commit assistant powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI. Parses git diff structures and creates beautifully formatted commit messages.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_GIT_COPILOT_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_GIT_COPILOT_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_GIT_COPILOT_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMGitCopilot] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("git_copilot", {})
        config_path = model_config.get("path", "../../models/qwen2.5-1.5b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX")
        print(f"[SLMGitCopilot] ONNX Model not found at configured path. Auto-downloading...")
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

    def generate_commit_message(self, diff_text: str, stream: bool = False):
        """
        Generates a Conventional Commit message based on a raw git diff text block.
        Truncates input if it exceeds reasonable context capacities.
        """
        if len(diff_text) > 4000:
            diff_text = diff_text[:4000] + "\n... (diff truncated for SLM context window optimization) ..."

        system_prompt = (
            "You are an expert Git copilot.\n"
            "Analyze the given git diff and output ONLY a beautiful conventional commit message. "
            "Use the exact template:\n"
            "<type>(<scope>): <short description>\n\n"
            "[optional longer body details]\n\n"
            "Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.\n"
            "Do not think out loud or output any other text or wrapping tags. Write the final commit message directly."
        )

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Git diff:\n{diff_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_tokens = self.tokenizer.encode(full_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + 512, temperature=0.0)

        if stream:
            def _stream_generator():
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        yield self.tokenizer.decode(new_tokens)
            return _stream_generator()

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        response_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                response_text += self.tokenizer.decode(new_tokens)

        return response_text
