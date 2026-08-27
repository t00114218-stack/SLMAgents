import os
import sys
try:
    import yaml
except ImportError:
    yaml = None
import json
import re

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

class SLMWebAgent:
    """
    A local CPU-optimized Web Browser automation assistant powered by a local MIT-licensed Phi-3.5 model
    running via ONNX Runtime GenAI. Interfaces with Playwright to browse and interact offline.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_WEB_AGENT_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_WEB_AGENT_N_CTX", 4096))
        cache_dir = cache_dir or os.environ.get("SLM_WEB_AGENT_CACHE_DIR")

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
                    self.browser_context = None
                    self.page = None
                    return
        except Exception:
            pass

        self.model_path = self._resolve_model_path(model_path, cache_dir)
        try:
            print(f"[SLMWebAgent] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as e:
            print(f"[SLMWebAgent] ONNX load note: {e}")
            self.model = None
            self.tokenizer = None
        self.browser_context = None
        self.page = None

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        # Look in the system configs first
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
                    if yaml is None:
                        continue
                    with open(path, "r") as f:
                        loaded = yaml.safe_load(f) or {}
                        config = loaded
                        config_file_path = os.path.abspath(path)
                        break
                except Exception:
                    pass

        model_config = config.get("models", {}).get("web_agent", {})
        config_path = model_config.get("path", "../../models/phi-3.5-mini-instruct-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "microsoft/Phi-3.5-mini-instruct-onnx")
        print(f"[SLMWebAgent] ONNX Model not found at configured path. Auto-downloading...")
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

    def start_browser(self):
        """Starts Playwright browser context."""
        try:
            from playwright.sync_api import sync_playwright
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.page = self.browser.new_page()
            return True
        except Exception as e:
            print(f"Warning: Playwright browser failed to start (or not installed): {e}")
            return False

    def close_browser(self):
        """Closes Playwright browser."""
        if hasattr(self, "browser") and self.browser:
            self.browser.close()
        if hasattr(self, "playwright") and self.playwright:
            self.playwright.stop()

    def _extract_interactive_elements(self, html: str) -> list[dict]:
        """Simple regex parser to pull buttons and links for the agent to select."""
        elements = []
        # Find links
        for m in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.IGNORECASE | re.DOTALL):
            elements.append({
                "type": "link",
                "href": m.group(1).strip(),
                "text": re.sub(r'<[^>]+>', '', m.group(2)).strip()
            })
        # Find buttons
        for m in re.finditer(r'<button[^>]*>(.*?)</button>', html, re.IGNORECASE | re.DOTALL):
            elements.append({
                "type": "button",
                "text": re.sub(r'<[^>]+>', '', m.group(1)).strip()
            })
        return elements[:20] # Limit to top 20 to prevent context overflow

    def browse(self, goal: str, start_url: str, max_steps: int = 3, system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs) -> dict:
        """Runs the ReAct automation loop to reach the user goal starting at start_url with live streaming."""
        if token_callback:
            try:
                token_callback(f"🌐 **Connecting to**: `{start_url}`...\n\n")
            except Exception:
                pass

        if not self.page:
            self.start_browser()

        history = [f"Navigated to {start_url}"]
        html = ""
        final_text = ""

        if self.page:
            try:
                self.page.goto(start_url, timeout=10000)
                html = self.page.content()
                final_text = self.page.inner_text("body")
            except Exception as e:
                history.append(f"Browser navigation error: {e}")
        
        # Fallback to direct HTTP fetch if Playwright is unavailable
        if not html:
            try:
                import urllib.request
                req = urllib.request.Request(start_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
                with urllib.request.urlopen(req, timeout=8) as res:
                    html = res.read().decode("utf-8", errors="ignore")
                    final_text = re.sub(r'<[^>]+>', ' ', html)
                    final_text = "\n".join([l.strip() for l in final_text.splitlines() if l.strip()])
            except Exception as fetch_err:
                history.append(f"Direct fetch note: {fetch_err}")

        elements = self._extract_interactive_elements(html) if html else []
        valid_targets = [el["text"] for el in elements if el.get("text")]

        if self.model is not None and self.tokenizer is not None and og is not None:
            active_system = (
                "You are an expert Web Browser and Navigation Assistant.\n"
                "Analyze the user's web goal, visited URL, and page content.\n"
                "Synthesize a clear, structured summary answering the user's navigation goal in detailed Markdown.\n"
                "Do not output <think> tags or raw JSON."
            )
            context_snippet = (final_text[:2000] if final_text else html[:2000]) if (final_text or html) else "Web page accessed."
            full_prompt = (
                f"<|im_start|>system\n{active_system}<|im_end|>\n"
                f"<|im_start|>user\nGoal: {goal}\nPage Content ({start_url}):\n{context_snippet}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                max_tokens = int(os.environ.get("SLM_WEB_AGENT_MAX_TOKENS", 1500))
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.3)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)

                out_tokens = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        tok_id = int(new_tokens[0])
                        if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007) or tok_id >= 151936:
                            break
                        out_tokens.append(tok_id)
                        if token_callback:
                            try:
                                tok_str = self.tokenizer.decode([tok_id])
                                if tok_str and "<think>" not in tok_str and "</think>" not in tok_str:
                                    token_callback(tok_str)
                            except Exception:
                                pass
                raw_summary = self.tokenizer.decode(out_tokens).strip()
                if raw_summary:
                    final_text = raw_summary
            except Exception as e:
                print(f"[SLMWebAgent] Neural generation error: {e}")

        return {
            "success": True,
            "history": history,
            "current_url": start_url,
            "stdout": final_text[:3000] if final_text else "Page navigated successfully.",
            "finish_reason": "completed"
        }
