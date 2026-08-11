import os
import sys
import yaml
import json
import re
from bs4 import BeautifulSoup

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

class SLMWebScraper:
    """
    A local CPU-optimized Web Scraper powered by a local MIT-licensed Phi-3.5 model
    running via ONNX Runtime GenAI. Cleans raw HTML inputs locally and extracts schema-defined structures.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_WEB_SCRAPER_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_WEB_SCRAPER_N_CTX", 4096))
        cache_dir = cache_dir or os.environ.get("SLM_WEB_SCRAPER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMWebScraper] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        target_path = None
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            target_path = os.path.abspath(model_path)
        else:
            # Config loading fallback
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

            model_config = config.get("models", {}).get("web_scraper", {})
            config_path = model_config.get("path", "../../models/phi-3.5-mini-instruct-onnx")
            config_path = os.path.expanduser(config_path)
            
            if not os.path.isabs(config_path) and config_file_path:
                config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
            target_path = config_path

        # Find directory containing genai_config.json
        for root, dirs, files in os.walk(target_path):
            if "genai_config.json" in files:
                return root
                
        if model_path:
            return target_path
            
        repo_id = model_config.get("repo_id", "microsoft/Phi-3.5-mini-instruct-onnx")
        print(f"[SLMWebScraper] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(target_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        for root, dirs, files in os.walk(target_path):
            if "genai_config.json" in files:
                return root
                
        return target_path

    def clean_html(self, html_content: str) -> str:
        """Removes script, style, and navigation tags from HTML to optimize input context window."""
        soup = BeautifulSoup(html_content, "lxml")
        
        # Remove non-content tags
        non_content_tags = ["script", "style", "nav", "footer", "header", "noscript", "aside", "iframe"]
        for element in soup(non_content_tags):
            element.extract()
            
        # Clean elements by classes or IDs matching navigation/advertising terms
        for element in soup.find_all(True):
            cls = element.get("class", [])
            if isinstance(cls, list):
                cls = " ".join(cls)
            cls = str(cls).lower()
            element_id = str(element.get("id", "")).lower()
            
            if any(term in cls or term in element_id for term in ["menu", "nav", "sidebar", "ad-", "banner", "footer"]):
                element.extract()
                
        # Get clean text representation
        text = soup.get_text(separator="\n")
        
        # Strip redundant white lines
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _get_vision_parser(self):
        if not hasattr(self, "_vision_parser") or self._vision_parser is None:
            try:
                from slm_vision_parser.vision_parser import SLMVisionParser  # type: ignore
            except ImportError:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                workspace_dir = os.path.dirname(base_dir)
                vision_parser_path = os.path.join(workspace_dir, "slm_vision_parser")
                if vision_parser_path not in sys.path:
                    sys.path.insert(0, vision_parser_path)
                from slm_vision_parser.vision_parser import SLMVisionParser  # type: ignore
            
            self._vision_parser = SLMVisionParser()
        return self._vision_parser

    def process_images_in_html(self, html_content: str, base_url: str = None) -> str:
        """Locates all <img> tags, downloads them, describes them via SLMVisionParser, and replaces tags with descriptions."""
        soup = BeautifulSoup(html_content, "lxml")
        img_tags = soup.find_all("img")
        
        if not img_tags:
            return html_content
            
        try:
            vision = self._get_vision_parser()
        except Exception as e:
            print(f"[SLMWebScraper] Vision parser import failed: {e}. Skipping image extraction.")
            return html_content
            
        import urllib.request
        import urllib.parse
        import tempfile
        
        for img in img_tags:
            src = img.get("src", "")
            if not src:
                continue
                
            # Resolve relative URLs
            if base_url:
                src_url = urllib.parse.urljoin(base_url, src)
            else:
                src_url = src
                
            temp_path = None
            try:
                print(f"[SLMWebScraper] Extracting and parsing image: {src_url} ...")
                
                # Fetch image data
                if "slmagents.ai" in src_url or src_url.startswith("file://") or not src_url.startswith("http"):
                    # Check locally
                    basename = src_url.split("/")[-1]
                    local_path = f"/Users/revathysuryaprakash/Documents/SLMAgents/website/{basename}"
                    if not os.path.exists(local_path):
                        local_path = f"/Users/revathysuryaprakash/Documents/SLMAgentsPortal/{basename}"
                    if os.path.exists(local_path):
                        temp_path = local_path
                else:
                    headers = {"User-Agent": "Mozilla/5.0"}
                    req = urllib.request.Request(src_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=8) as response:
                        img_data = response.read()
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(img_data)
                        temp_path = f.name
                        
                if temp_path and os.path.exists(temp_path):
                    # Run vision parser
                    desc = vision.parse_image(temp_path, task="<DETAILED_CAPTION>")
                    if desc.strip():
                        desc_text = f" [Image Description: {desc.strip()}] "
                        img.replace_with(soup.new_string(desc_text))
            except Exception as e:
                print(f"[SLMWebScraper] Failed to parse image {src}: {e}")
            finally:
                # Only remove downloaded temp files
                if temp_path and temp_path.startswith(tempfile.gettempdir()) and os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        return str(soup)

    def describe_tables_in_html(self, html_content: str) -> str:
        """Finds all <table> elements and describes them using the local Phi-3.5 model, replacing them in-place."""
        soup = BeautifulSoup(html_content, "lxml")
        tables = soup.find_all("table")
        
        if not tables:
            return html_content
            
        for table in tables:
            table_str = str(table)
            
            # Phi-3.5 instructions for translating tabular DOM tree to natural text paragraph
            system_prompt = (
                "You are an offline HTML table explainer.\n"
                "Read the HTML table and generate a natural language text description summarizing the columns, rows, entries, and parameters clearly. Do not output markdown tables or pipes."
            )
            user_prompt = f"HTML Table:\n{table_str}\n\nGenerate a clean natural language paragraph describing this table:"
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            full_prompt = ""
            for msg in messages:
                full_prompt += f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
            full_prompt += "<|assistant|>\n"
            
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + 512, temperature=0.0)
                
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                response_text = ""
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        response_text += self.tokenizer.decode(new_tokens)
                
                desc = response_text.strip()
                if desc:
                    table.replace_with(soup.new_string(f" [Table Description: {desc}] "))
            except Exception as e:
                print(f"[SLMWebScraper] Table description failed: {e}")
                
        return str(soup)

    def scrape_url(self, url: str, schema_dict: dict = None, max_retries: int = 3):
        """Fetches html from URL and extracts cleaned main content text, avoiding navigation/menus and ads."""
        if "slmagents.ai" in url or "localhost" in url:
            basename = url.split("/")[-1]
            if not basename:
                basename = "index.html"
            local_path = f"/Users/revathysuryaprakash/Documents/SLMAgents/website/{basename}"
            if not os.path.exists(local_path):
                local_path = f"/Users/revathysuryaprakash/Documents/SLMAgentsPortal/{basename}"
            
            with open(local_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        else:
            import urllib.request
            headers = {"User-Agent": "Mozilla/5.0"}
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                html_content = response.read().decode("utf-8", errors="ignore")
                
        html_content = self.process_images_in_html(html_content, base_url=url)
        html_content = self.describe_tables_in_html(html_content)
        if schema_dict is None:
            return self.clean_html(html_content)
            
        return self.scrape(html_content, schema_dict, max_retries)

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
        return text.strip()

    def scrape(self, html_content: str, schema_dict: dict, max_retries: int = 3) -> dict:
        """Strips HTML content and parses the remainder into a schema compliant JSON structure."""
        html_content = self.process_images_in_html(html_content)
        html_content = self.describe_tables_in_html(html_content)
        cleaned_text = self.clean_html(html_content)
        
        system_prompt = (
            "You are a local Web Scraper utility.\n"
            "Analyze the cleaned web text and extract the data to populate a structured JSON block matching the target schema. "
            "IMPORTANT: Output actual extracted data. Never copy schema type descriptors (such as 'string', 'integer', 'boolean') or templates. "
            "Return the final completed JSON inside a ```json ... ``` code block. Never output explanation headers or footers."
        )

        user_prompt = (
            f"Web Page Content:\n{cleaned_text[:8000]}\n\n"
            f"Target JSON Schema Structure to Populate:\n{json.dumps(schema_dict, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            full_prompt = ""
            for msg in messages:
                full_prompt += f"<|{msg['role']}|>\n{msg['content']}<|end|>\n"
            full_prompt += "<|assistant|>\n"

            input_tokens = self.tokenizer.encode(full_prompt)
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=len(input_tokens) + 1024, temperature=0.0)
            
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            
            response_text = ""
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    response_text += self.tokenizer.decode(new_tokens)

            json_block = self._extract_json(response_text)
            try:
                parsed = json.loads(json_block)
                return parsed
            except Exception as e:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"JSON syntax failed with error: {e}. Correct formatting errors and return valid schema block inside ```json ```."
                })

        return {"error": "Scraping failed to align with schema dict specifications"}
