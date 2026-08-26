import os
import sys
import yaml
import re
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
            
        try:
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if not main_mod or not hasattr(main_mod, "get_shared_onnx_genai"):
                try:
                    import importlib
                    main_mod = importlib.import_module("main")
                except Exception:
                    main_mod = None
            if main_mod and hasattr(main_mod, "get_shared_onnx_genai"):
                self.model, self.tokenizer = main_mod.get_shared_onnx_genai()
                if self.model and self.tokenizer:
                    self.model_path = "shared_onnx"
                    return
        except Exception:
            pass

        self.model_path = self._resolve_model_path(model_path, cache_dir)
        try:
            print(f"[SLMGitRepoManager] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as e:
            print(f"[SLMGitRepoManager] ONNX load note: {e}")
            self.model = None
            self.tokenizer = None
        
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path and os.path.exists(model_path):
            return os.path.abspath(model_path)

        shared_qwen = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "qwen3.5-0.8b-onnx")
        if os.path.exists(shared_qwen):
            return shared_qwen

        shared_phi = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "phi-3.5-mini-instruct-onnx", "cpu_and_mobile", "cpu-int4-awq-block-128-acc-level-4")
        if os.path.exists(shared_phi):
            return shared_phi

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("git_repo_manager", {})
        config_path = model_config.get("path", "../../models/qwen3.5-0.8b-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
                
        return shared_phi if os.path.exists(shared_phi) else config_path

    def _clean_text(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        return text.strip()

    def process_repo_request(self, query: str = "", diff_text: str = "", system_prompt: str = None, token_callback: callable = None, **kwargs) -> str:
        """
        Dynamically analyzes git repositories, commit logs, merge conflicts, release notes, and diffs using the local SLM.
        """
        self._lazy_init_onnx()

        req_text = (query or diff_text or "").strip()
        if not req_text:
            return "Please provide a git diff, branch question, commit log, or release notes requirement."

        # If it's strictly a git diff, generate conventional commit message
        if diff_text and ("diff --git" in diff_text or "@@ " in diff_text or "--- a/" in diff_text):
            return self.generate_commit_message(diff_text=diff_text, system_prompt=system_prompt, token_callback=token_callback)

        default_sys = (
            "You are a Principal Release Engineer and Git Repository Manager.\n"
            "Analyze the repository request thoroughly and provide a structured, actionable Git operations and release engineering report.\n\n"
            "Structure your response with clear Markdown headings:\n"
            "### 1. Commit History & Architectural Impact\n"
            "Evaluate recent changes, development velocity, and affected subsystems.\n\n"
            "### 2. Cross-Branch Merge Conflict Risk Assessment\n"
            "Identify potential merge conflicts, shared file collision risks across branches, and branch divergence mitigation strategies.\n\n"
            "### 3. Production Release Notes\n"
            "Draft structured release notes categorized by Features, Bug Fixes, Performance Improvements, and Breaking Changes.\n\n"
            "### 4. Git Execution Playbook\n"
            "Provide exact, copy-pasteable Git CLI commands in a ```bash ``` code block for branching, tagging, and deployment."
        )
        active_sys = system_prompt or default_sys

        if self.model is not None and self.tokenizer is not None and og is not None:
            prompt = (
                "<|im_start|>system\n"
                f"{active_sys}<|im_end|>\n"
                f"<|im_start|>user\n{req_text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + 700, temperature=0.3)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)

                tokens_out = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        tok_id = int(new_tokens[0])
                        if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        tokens_out.append(tok_id)
                        if token_callback:
                            tok_str = self.tokenizer.decode([tok_id])
                            token_callback(tok_str)
                res_text = self.tokenizer.decode(tokens_out).strip()
                if "<|im_end|>" in res_text:
                    res_text = res_text.replace("<|im_end|>", "").strip()
                if res_text:
                    return res_text
            except Exception as e:
                print(f"[SLMGitRepoManager] Generation error: {e}")
                return f"Error analyzing git repository: {e}"

        return "Git repository management model is initializing. Please try again in a moment."

    def _lazy_init_onnx(self):
        try:
            main_mod = sys.modules.get("main") or sys.modules.get("__main__")
            if not main_mod or not hasattr(main_mod, "get_shared_onnx_genai"):
                try:
                    import importlib
                    main_mod = importlib.import_module("main")
                except Exception:
                    main_mod = None
            if main_mod and hasattr(main_mod, "get_shared_onnx_genai"):
                m, tok = main_mod.get_shared_onnx_genai()
                if m and tok:
                    self.model = m
                    self.tokenizer = tok
        except Exception:
            pass

    def generate_commit_message(self, diff_text: str, stream: bool = False, system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs):
        self._lazy_init_onnx()
        if not self.model or not self.tokenizer:
            return "feat: update codebase changes"

        if not diff_text or not diff_text.strip():
            return "chore: update repository files"

        system_prompt = system_prompt or (
            "You are an expert Git and version control engineer. Analyze the provided git diff and write a high quality, "
            "concise, conventional git commit message summarizing the changes according to the Conventional Commits specification.\n"
            "Format:\n<type>(<scope>): <short description>\n\n[optional longer body bullet points]\n"
            "Allowed types: feat, fix, docs, style, refactor, perf, test, build, ci, chore, revert.\n"
            "Do not think out loud or output any <think> tags. Write the final commit message directly."
        )

        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Git diff:\n{diff_text}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )

        input_tokens = self.tokenizer.encode(full_prompt)
        max_tokens = int(os.environ.get("SLM_GIT_REPO_MANAGER_MAX_TOKENS", 1000))
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.7)

        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        tokens_out = []
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                token_id = int(new_tokens[0])
                if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                    break
                tokens_out.append(token_id)
                if token_callback:
                    token_callback(self.tokenizer.decode([token_id]))

        raw_msg = self.tokenizer.decode(tokens_out).strip()
        if "<|im_end|>" in raw_msg:
            raw_msg = raw_msg.replace("<|im_end|>", "").strip()
        return self._clean_text(raw_msg)

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
        max_tokens = int(os.environ.get("SLM_GIT_REPO_MANAGER_MAX_TOKENS", 3000))
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.7)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        resolved_text = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                token_id = int(new_tokens[0])
                if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                    break
                resolved_text += self.tokenizer.decode(new_tokens)
                
        return resolved_text.strip()
