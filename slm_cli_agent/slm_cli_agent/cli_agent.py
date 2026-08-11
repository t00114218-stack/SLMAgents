import os
import sys
import yaml
import re
import subprocess

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_CLI_AGENT_CONFIG"),
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

class SLMCLIAgent:
    """
    A local CPU-optimized CLI companion agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI. Recommends, explains, and safely executes system shell commands.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_CLI_AGENT_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_CLI_AGENT_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_CLI_AGENT_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMCLIAgent] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("cli_agent", {})
        config_path = model_config.get("path", "../../models/qwen2.5-1.5b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX")
        print(f"[SLMCLIAgent] ONNX Model not found at configured path. Auto-downloading...")
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

    def _extract_command(self, text: str) -> str:
        match = re.search(r"```bash\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_sh = re.search(r"```sh\s*(.*?)\s*```", text, re.DOTALL)
        if match_sh:
            return match_sh.group(1).strip()
        return ""

    def generate_command(self, query: str, stream: bool = False):
        """
        Translates a natural language request to a command sequence.
        If stream=True, returns a generator of tokens.
        If stream=False, runs to completion and returns the final (command, full_response) tuple.
        """
        system_prompt = (
            "You are a local shell automation CLI helper.\n"
            "Analyze the user's requirement, think inside <thought>...</thought> tags, "
            "and output the precise command wrapped inside a single ```bash ... ``` code block. "
            "Explain briefly what the command does, prioritizing non-destructive execution flag options."
        )

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Command request: {query}<|im_end|>\n"
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

        command = self._extract_command(response_text)
        return command, response_text

    def execute_command(self, cmd: str) -> tuple[int, str, str]:
        """
        Safely executes the proposed shell command locally with built-in defense sequences.
        """
        # Strict prevention of dangerous system-level changes
        dangerous = ["rm -rf /", "mkfs", "dd if=", "shutdown", "reboot"]
        if any(d in cmd for d in dangerous):
            return -1, "", "Execution Blocked: Destructive or dangerous command pattern detected."

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=20.0
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command Execution Timed Out after 20 seconds."
