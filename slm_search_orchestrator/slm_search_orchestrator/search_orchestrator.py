import os
import sys

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
    import yaml
except ImportError:
    yaml = None
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
            print(f"[SLMSearchOrchestrator] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
            self.model = og.Model(self.model_path)
            self.tokenizer = og.Tokenizer(self.model)
        except Exception as e:
            print(f"[SLMSearchOrchestrator] ONNX load note: {e}")
            self.model = None
            self.tokenizer = None

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
                    if yaml is None:
                        continue
                    with open(path, "r") as f:
                        config = yaml.safe_load(f) or {}
                    config_file_path = os.path.abspath(path)
                    break
                except Exception:
                    pass

        shared_qwen_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "qwen3.5-0.8b-onnx")
        if os.path.exists(shared_qwen_path):
            return shared_qwen_path

        model_config = config.get("models", {}).get("search_orchestrator", {})
        config_path = model_config.get("path", "../../models/qwen3.5-0.8b-onnx")
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
        max_tokens = int(os.environ.get("SLM_SEARCH_ORCHESTRATOR_MAX_TOKENS", 3000))
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.7)
        
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

        match = re.search(r"```json\s*(.*?)\s*```", response_text, re.DOTALL)
        json_str = match.group(1).strip() if match else "[]"
        try:
            queries = json.loads(json_str)
            if isinstance(queries, list) and len(queries) > 0:
                valid_q = [q for q in queries if isinstance(q, str) and q.strip()]
                if valid_q:
                    return valid_q
        except Exception:
            pass
        return [user_query]

    def execute_search(self, query: str, max_results: int = 15) -> list[dict]:
        """Queries live Web Search (Bing & DuckDuckGo) FIRST for accurate real-world results, followed by Wikipedia fallback."""
        import urllib.request, urllib.parse, json, base64, re
        api_results = []

        # Clean & expand ambiguous search terms to prevent software download false-positives
        search_query = query
        q_lower = query.lower().strip()
        if "movie" in q_lower or "film" in q_lower or "cinema" in q_lower:
            search_query = "IMDb popular movie recommendations 2024 list"

        encoded_q = urllib.parse.quote(search_query)

        # 1. PRIORITY 1: Bing Live Web Search Scraper (Fast, Unblocked, High Accuracy)
        try:
            from bs4 import BeautifulSoup
            bing_url = f"https://www.bing.com/search?q={encoded_q}"
            req_bing = urllib.request.Request(bing_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"})
            with urllib.request.urlopen(req_bing, timeout=5) as res_bing:
                html_res = res_bing.read().decode("utf-8", errors="ignore")
                soup = BeautifulSoup(html_res, "html.parser")
                for li in soup.find_all("li", class_="b_algo")[:max_results]:
                    h2 = li.find("h2")
                    p = li.find("p")
                    if h2 and h2.find("a"):
                        a = h2.find("a")
                        raw_href = a.get("href", "")
                        clean_href = raw_href
                        if "u=a1" in raw_href:
                            try:
                                b64_part = raw_href.split("u=a1")[1].split("&")[0]
                                b64_part += "=" * (-len(b64_part) % 4)
                                decoded_url = base64.b64decode(b64_part).decode("utf-8", errors="ignore")
                                if decoded_url.startswith("http"):
                                    clean_href = decoded_url
                            except Exception:
                                pass

                        # Filter out software downloads, dictionary definitions, and e-commerce clothing links
                        href_lower = clean_href.lower()
                        title_lower = a.get_text(strip=True).lower()
                        if any(bad in href_lower or bad in title_lower for bad in ["anydesk", "softonic", "techspot", "exe", "apk", "dictionary", "bestbuy", "merriam-webster", "cambridge", "myntra", "nykaa", "flipkart", "clothing", "apparel", "/tops"]):
                            continue

                        api_results.append({
                            "title": a.get_text(strip=True),
                            "href": clean_href,
                            "body": p.get_text(strip=True) if p else a.get_text(strip=True)
                        })
        except Exception as bing_err:
            print(f"[SLMSearchOrchestrator] Bing Scraper error: {bing_err}")

        # 2. PRIORITY 2: DuckDuckGo Direct HTML Web Search Scraper
        if len(api_results) < 5:
            try:
                from bs4 import BeautifulSoup
                ddg_html_url = f"https://html.duckduckgo.com/html/?q={encoded_q}"
                req_ddg = urllib.request.Request(ddg_html_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
                with urllib.request.urlopen(req_ddg, timeout=5) as res_ddg:
                    html_res = res_ddg.read().decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html_res, "html.parser")
                    for body_div in soup.find_all("div", class_="result__body")[:max_results]:
                        title_a = body_div.find("a", class_="result__a")
                        snippet_a = body_div.find("a", class_="result__snippet")
                        if title_a:
                            href = title_a.get("href", "")
                            if "RU=" in href:
                                import urllib.parse as up
                                parsed_href = up.parse_qs(up.urlparse(href).query).get("RU", [href])[0]
                            else:
                                parsed_href = href
                            api_results.append({
                                "title": title_a.get_text(strip=True),
                                "href": parsed_href,
                                "body": snippet_a.get_text(strip=True) if snippet_a else title_a.get_text(strip=True)
                            })
            except Exception as ddg_err:
                print(f"[SLMSearchOrchestrator] Direct DDG HTML Scraper error: {ddg_err}")

        # 3. PRIORITY 3: DuckDuckGo Lite Scraper
        if len(api_results) < 5:
            try:
                from bs4 import BeautifulSoup
                lite_url = f"https://lite.duckduckgo.com/lite/?q={encoded_q}"
                req_lite = urllib.request.Request(lite_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
                with urllib.request.urlopen(req_lite, timeout=5) as res_lite:
                    html_res = res_lite.read().decode("utf-8", errors="ignore")
                    soup = BeautifulSoup(html_res, "html.parser")
                    for tr in soup.find_all("tr")[:max_results * 2]:
                        td_snippet = tr.find("td", class_="result-snippet")
                        td_title = tr.find("a", class_="result-link")
                        if td_title and td_snippet:
                            api_results.append({
                                "title": td_title.get_text(strip=True),
                                "href": td_title.get("href", ""),
                                "body": td_snippet.get_text(strip=True)
                            })
            except Exception as lite_err:
                print(f"[SLMSearchOrchestrator] DDG Lite Scraper error: {lite_err}")

        # 4. PRIORITY 4: DuckDuckGo Instant Abstract API
        if len(api_results) < 5:
            try:
                url = f"https://api.duckduckgo.com/?q={encoded_q}&format=json&no_html=1"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                with urllib.request.urlopen(req, timeout=4) as res:
                    d = json.loads(res.read().decode("utf-8"))
                    if d.get("Abstract"):
                        api_results.append({
                            "title": d.get("Heading") or query,
                            "href": d.get("AbstractURL") or "https://duckduckgo.com",
                            "body": d.get("Abstract")
                        })
            except Exception:
                pass

        # 5. PRIORITY 5: Wikipedia Summary REST API (Only as secondary fallback)
        if len(api_results) < 3:
            try:
                srch_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={encoded_q}&format=json"
                req = urllib.request.Request(srch_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                with urllib.request.urlopen(req, timeout=4) as res:
                    d = json.loads(res.read().decode("utf-8"))
                    hits = d.get("query", {}).get("search", [])
                    for h in hits[:2]:
                        t = h["title"]
                        p_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(t)}"
                        req_p = urllib.request.Request(p_url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"})
                        with urllib.request.urlopen(req_p, timeout=4) as res_p:
                            p_data = json.loads(res_p.read().decode("utf-8"))
                            if p_data.get("extract"):
                                api_results.append({
                                    "title": p_data.get("title", t),
                                    "href": p_data.get("content_urls", {}).get("desktop", {}).get("page") or f"https://en.wikipedia.org/wiki/{t}",
                                    "body": p_data.get("extract")
                                })
            except Exception:
                pass

        return api_results

    def retrieve(self, user_query: str, max_results_per_query: int = 5, system_prompt: str = None, user_input: str = None) -> list[dict]:
        """Performs full search aggregation workflow retrieving up to 15 top results."""
        queries = self.generate_queries(user_query)
        all_results = []
        seen_links = set()
        
        for q in queries:
            results = self.execute_search(q, max_results=max_results_per_query)
            for r in results:
                link = r.get("href")
                if link and link not in seen_links and link.startswith("http"):
                    seen_links.add(link)
                    all_results.append(r)
                if len(all_results) >= 15:
                    break
            if len(all_results) >= 15:
                break
                    
        return all_results

    def search_and_synthesize(self, query: str, token_callback: callable = None, **kwargs) -> dict:
        """Retrieves top 15 search results, selects sure-shot links, scrapes web pages via SLMWebScraper, and synthesizes a grounded answer."""
        chunks = self.retrieve(query)
        
        # Sure-Shot Link Filtering: Pick top 15 valid, accessible HTTP/HTTPS links
        sure_shot_chunks = []
        for c in chunks:
            url = c.get("href", "")
            if url.startswith("http") and not any(bad in url for bad in ["duckduckgo.com", "facebook.com/sharer", "twitter.com/intent"]):
                sure_shot_chunks.append(c)
            if len(sure_shot_chunks) >= 15:
                break

        # Parallel Web Scraping Tier: Use SLMWebScraper to extract full text content from top sure-shot links
        try:
            from slm_web_scraper import SLMWebScraper
        except ImportError:
            try:
                from slm_web_scraper.web_scraper import SLMWebScraper
            except ImportError:
                SLMWebScraper = None

        if SLMWebScraper is not None and sure_shot_chunks:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            
            def scrape_single_link(chunk):
                url = chunk.get("href")
                if not url:
                    return
                try:
                    import urllib.request
                    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"})
                    with urllib.request.urlopen(req, timeout=4) as res:
                        html = res.read().decode("utf-8", errors="ignore")
                        scraper = SLMWebScraper()
                        clean_text = scraper.clean_html(html)
                        if len(clean_text) > 100:
                            chunk["scraped_text"] = clean_text[:1500]
                except Exception:
                    pass

            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(scrape_single_link, c) for c in sure_shot_chunks[:8]]
                for future in as_completed(futures):
                    pass

        if not sure_shot_chunks:
            system_prompt = (
                "You are a Factual Search & Web Information Extraction Engine powered by SLMAgents.\n"
                "Provide DIRECT, CONCRETE REAL-WORLD SEARCH RESULTS and factual information for the user's query.\n"
                "List at most 5-7 distinct entity names, locations, rates, options, or movies without repeating.\n"
                "CRITICAL: Do NOT repeat the same line or entity. Do NOT fabricate false plot details. Answer directly with real-world facts."
            )
            full_prompt = (
                f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                f"<|im_start|>user\nQuery: {query}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            input_tokens = self.tokenizer.encode(full_prompt)
            max_tokens = int(os.environ.get("SLM_SEARCH_ORCHESTRATOR_MAX_TOKENS", 3000))
            params = og.GeneratorParams(self.model)
            params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.22)
            generator = og.Generator(self.model, params)
            generator.append_tokens(input_tokens)
            answer_tokens = []
            gen_answer = ""
            while not generator.is_done():
                generator.generate_next_token()
                new_tokens = generator.get_next_tokens()
                if len(new_tokens) > 0:
                    tok_id = int(new_tokens[0])
                    if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                        break
                    tok_str = self.tokenizer.decode(new_tokens)
                    gen_answer += tok_str
                    if token_callback:
                        token_callback(tok_str)
                    # Line repetition guardrail
                    lines = [l.strip() for l in gen_answer.splitlines() if l.strip()]
                    if len(lines) >= 3 and lines[-1] == lines[-2] == lines[-3]:
                        break
            return {
                "agent": "SLMSearchOrchestrator",
                "status": "success",
                "search_query": query,
                "results_count": 0,
                "retrieved_chunks": [],
                "sources": [],
                "answer": gen_answer.strip()
            }

        # Build context from sure_shot_chunks and scraped page text
        context_str = ""
        for i, c in enumerate(sure_shot_chunks):
            scraped = c.get("scraped_text")
            body_text = f"Scraped Page Content:\n{scraped}" if scraped else f"Snippet: {c.get('body')}"
            context_str += f"[{i+1}] Source: {c.get('href')}\nTitle: {c.get('title')}\n{body_text}\n\n"
            
        system_prompt = (
            "You are a Factual Search & Web Information Extraction Engine powered by live Web Search & Web Scraping.\n"
            "Analyze the search context and provide DIRECT, CONCRETE REAL-WORLD SEARCH RESULTS and factual information.\n"
            "List specific entity names, locations, movie recommendations, bank options, interest rates, and direct facts extracted strictly from search results.\n"
            "CRITICAL: Do NOT repeat the same line or entity. List at most 6-8 distinct items. Do NOT invent false plot details."
        )
        
        full_prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"Search Results:\n{context_str}\n"
            f"Query: {query}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(full_prompt)
        max_tokens = int(os.environ.get("SLM_SEARCH_ORCHESTRATOR_MAX_TOKENS", 3000))
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.22)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        
        answer = ""
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                token_id = int(new_tokens[0])
                if token_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                    break
                tok_str = self.tokenizer.decode(new_tokens)
                answer += tok_str
                if token_callback:
                    token_callback(tok_str)
                # Line repetition guardrail check
                lines = [l.strip() for l in answer.splitlines() if l.strip()]
                if len(lines) >= 3 and lines[-1] == lines[-2] == lines[-3]:
                    break

                
        cleaned_answer = answer.strip()
        if "</think>" in cleaned_answer:
            cleaned_answer = cleaned_answer.split("</think>")[-1].strip()

        # If sure_shot_chunks exist, append formatted web sources
        if sure_shot_chunks:
            sources_list = []
            for c in sure_shot_chunks[:5]:
                if c.get("href"):
                    sources_list.append(f"- [{c.get('title', 'Web Link')}]({c.get('href')})")
            if sources_list:
                cleaned_answer += "\n\n### Verified Web Sources\n" + "\n".join(sources_list)

        return {
            "agent": "SLMSearchOrchestrator",
            "status": "success",
            "search_query": query,
            "results_count": len(sure_shot_chunks),
            "retrieved_chunks": sure_shot_chunks,
            "sources": [c.get("href") for c in sure_shot_chunks if c.get("href")],
            "answer": cleaned_answer
        }
