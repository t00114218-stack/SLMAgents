import os
import sys
import yaml

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_GIT_REPO_MANAGER_CONFIG"),
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

class SLMGitRepoManager:
    """
    A local CPU-optimized Conventional Commit and Git repository manager powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI. Parses git diff structures and creates beautifully formatted commit messages.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_GIT_REPO_MANAGER_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_GIT_REPO_MANAGER_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_GIT_REPO_MANAGER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMGitRepoManager] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("git_repo_manager", {})
        config_path = model_config.get("path", "../../models/qwen2.5-1.5b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX")
        print(f"[SLMGitRepoManager] ONNX Model not found at configured path. Auto-downloading...")
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

    def generate_commit_message(self, diff_text: str, stream: bool = False, system_prompt: str = None, user_input: str = None):
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

    def commit(self, message: str = None) -> tuple[bool, str]:
        """
        Stages all tracked modified changes, generates a commit message, and commits.
        """
        import subprocess
        subprocess.run("git add -u", shell=True)
        
        diff_res = subprocess.run("git diff --staged", shell=True, capture_output=True, text=True)
        if not diff_res.stdout.strip():
            return False, "No staged modifications found to commit."
            
        if not message:
            message = self.generate_commit_message(diff_res.stdout)
            
        commit_res = subprocess.run(f"git commit -m '{message}'", shell=True, capture_output=True, text=True)
        if commit_res.returncode == 0:
            return True, f"Committed successfully with message:\n{message}"
        return False, f"Git commit failed:\n{commit_res.stderr}"

    def merge(self, branch: str) -> tuple[bool, str]:
        """
        Merges the specified branch. Detects if conflicts occur.
        """
        import subprocess
        res = subprocess.run(f"git merge {branch}", shell=True, capture_output=True, text=True)
        if res.returncode == 0:
            return True, f"Merged branch '{branch}' successfully."
            
        conflict_res = subprocess.run("git diff --name-only --diff-filter=U", shell=True, capture_output=True, text=True)
        conflicted_files = conflict_res.stdout.strip().splitlines()
        if conflicted_files:
            return False, f"Merge conflicts detected in: {', '.join(conflicted_files)}"
            
        return False, f"Merge failed:\n{res.stderr}"

    def resolve_conflicts(self) -> dict:
        """
        Scans for conflict markers, asks the SLM to merge conflict hunks, rewrites files, and stages them.
        """
        import subprocess
        import re
        conflict_res = subprocess.run("git diff --name-only --diff-filter=U", shell=True, capture_output=True, text=True)
        conflicted_files = conflict_res.stdout.strip().splitlines()
        if not conflicted_files:
            return {"success": True, "details": "No merge conflicts to resolve."}
            
        resolved = []
        failed = []
        
        for filepath in conflicted_files:
            if not os.path.exists(filepath):
                continue
            try:
                with open(filepath, "r") as f:
                    content = f.read()
                
                # Check for standard conflict markers
                pattern = r"<<<<<<< (.*?)\n(.*?)\n=======\n(.*?)\n>>>>>>> (.*?)\n"
                matches = list(re.finditer(pattern, content, re.DOTALL))
                if not matches:
                    continue
                    
                new_content = content
                for m in reversed(matches):
                    full_match = m.group(0)
                    local_code = m.group(2)
                    incoming_code = m.group(3)
                    
                    resolution = self._ask_model_to_resolve(filepath, local_code, incoming_code)
                    new_content = new_content.replace(full_match, resolution + "\n")
                    
                with open(filepath, "w") as f:
                    f.write(new_content)
                
                subprocess.run(f"git add {filepath}", shell=True)
                resolved.append(filepath)
            except Exception as e:
                failed.append((filepath, str(e)))
                
        return {
            "success": len(failed) == 0,
            "resolved": resolved,
            "failed": failed
        }

    def _ask_model_to_resolve(self, filepath: str, local: str, incoming: str) -> str:
        """
        Internal assistant to combine conflicting hunks.
        """
        system_prompt = (
            "You are an expert developer resolving git merge conflicts.\n"
            "Combine the two code blocks logically, resolving any duplicate code or API parameters. "
            "Output ONLY the final resolved clean code. Do not include conflict markers or markdown wraps."
        )
        
        user_prompt = (
            f"File: {filepath}\n"
            f"--- LOCAL BLOCK ---\n{local}\n"
            f"--- INCOMING BLOCK ---\n{incoming}\n"
            "Resolved clean output code:"
        )
        
        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"{user_prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(full_prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + 1024, temperature=0.0)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        resolved_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                resolved_text += self.tokenizer.decode(new_tokens)
                
        return resolved_text.strip()
