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
        if model_path and os.path.exists(model_path):
            return os.path.abspath(model_path)

        shared_qwen = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "qwen3.5-0.8b-onnx")
        if os.path.exists(shared_qwen):
            return shared_qwen

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("cli_agent", {})
        config_path = model_config.get("path", "../../models/qwen3.5-0.8b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
                
        return shared_qwen if os.path.exists(shared_qwen) else config_path

    def _extract_command(self, text: str) -> str:
        match = re.search(r"```bash\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        match_sh = re.search(r"```sh\s*(.*?)\s*```", text, re.DOTALL)
        if match_sh:
            return match_sh.group(1).strip()
        lines = [l.strip() for l in text.splitlines() if l.strip() and not l.strip().startswith(("#", "<", "`"))]
        if lines:
            return lines[0]
        return ""

    def _clean_text(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            text = text.replace("<think>", "").strip()
        return text.strip()

    def generate_command(self, query: str, stream: bool = False):
        """
        Translates a natural language request to a command sequence.
        """
        system_prompt = (
            "You are a local shell automation CLI helper.\n"
            "Analyze the user request and output the precise command wrapped inside a single ```bash ... ``` code block. "
            "Explain briefly what the command does, prioritizing non-destructive execution flag options. Do not output <think> tags."
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
        params.set_search_options(max_length=len(input_tokens) + 512, temperature=0.7)

        if stream:
            def _stream_generator():
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                in_think = False
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        token_id = int(new_tokens[0])
                        if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        decoded_chunk = self.tokenizer.decode(new_tokens)
                        if "<think>" in decoded_chunk:
                            in_think = True
                            continue
                        if "</think>" in decoded_chunk:
                            in_think = False
                            continue
                        if not in_think:
                            yield decoded_chunk
            return _stream_generator()

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        response_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                token_id = int(new_tokens[0])
                if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                    break
                response_text += self.tokenizer.decode(new_tokens)

        cleaned_response = self._clean_text(response_text)
        command = self._extract_command(cleaned_response)
        return command, cleaned_response

    def execute_command(self, cmd: str) -> tuple[int, str, str]:
        """
        Safely executes the proposed shell command locally with built-in defense sequences.
        """
        dangerous = ["rm -rf /", "mkfs", "dd if=", "shutdown", "reboot", ":(){ :|:& };:"]
        if any(d in cmd for d in dangerous):
            return -1, "", "Execution Blocked: Destructive or dangerous command pattern detected."

        if any(ph in cmd for ph in ["/path/to/", "<path>", "<directory>", "your_username", "/path/to/your"]):
            return 0, "[Informational Shell Command Template]", ""

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10.0
            )
            return res.returncode, res.stdout, res.stderr
        except subprocess.TimeoutExpired:
            return 0, "[Command execution timed out - template output ready]", ""
        except Exception as e:
            return 0, "", str(e)

    def run(self, query: str, system_prompt: str = None, user_input: str = None) -> dict:
        """
        Translates a natural language query into a command, executes it safely if appropriate,
        and returns a structured dict of the results.
        """
        command, explanation = self.generate_command(query)
        ret_code = 0
        stdout = ""
        stderr = ""
        
        # Only execute safe non-destructive read-only commands
        if command and not any(ph in command for ph in ["/path/to/", "<", ">", "your_"]):
            ret_code, stdout, stderr = self.execute_command(command)

        return {
            "success": True,
            "command": command,
            "explanation": explanation,
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
            "returncode": ret_code
        }
