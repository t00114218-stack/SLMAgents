import os
import sys
import yaml
import json
import re
from pypdf import PdfReader
from docx import Document

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_DOCUMENT_PARSER_CONFIG"),
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

class SLMDocumentParser:
    """
    A local CPU-optimized Document Parser agent powered by a local MIT-licensed Phi-3.5 model
    running via ONNX Runtime GenAI. Extracts data from DOCX, PDF, and Markdown files into structured JSON schemas.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using:\n"
                "pip install onnxruntime-genai"
            )

        n_threads = n_threads or int(os.environ.get("SLM_DOCUMENT_PARSER_N_THREADS", 4))
        self.n_ctx     = n_ctx     or int(os.environ.get("SLM_DOCUMENT_PARSER_N_CTX", 4096))
        cache_dir = cache_dir or os.environ.get("SLM_DOCUMENT_PARSER_CACHE_DIR")

        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        print(f"[SLMDocumentParser] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)

    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("document_parser", {})
        config_path = model_config.get("path", "../../models/phi-3.5-mini-instruct-onnx")
        config_path = os.path.expanduser(config_path)
        
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        repo_id = model_config.get("repo_id", "microsoft/Phi-3.5-mini-instruct-onnx")
        print(f"[SLMDocumentParser] ONNX Model not found at configured path. Auto-downloading...")
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

    def extract_text(self, file_path: str) -> str:
        """Extracts raw text content from DOCX, PDF, or text documents."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == ".docx":
            doc = Document(file_path)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".pdf":
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        else:
            # Fallback to plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
        return text.strip()

    def parse(self, file_path: str, schema_dict: dict, max_retries: int = 3) -> dict:
        """Parses a document file into a structured JSON dict matching the schema_dict."""
        raw_text = self.extract_text(file_path)
        
        system_prompt = (
            "You are a local Document Parser agent.\n"
            "Analyze the document text and extract the information into a structured JSON block matching the target schema inside a ```json ... ``` code block.\n"
            "Never output explanations outside of the code block."
        )

        user_prompt = (
            f"Document Text:\n{raw_text[:8000]}\n\n"
            f"Required Target JSON Schema:\n{json.dumps(schema_dict, indent=2)}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        for attempt in range(max_retries):
            full_prompt = ""
            for msg in messages:
                full_prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
            full_prompt += "<|im_start|>assistant\n"

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
                    "content": f"JSON parsing failed with error: {e}. Correct the JSON format and return the complete updated block inside ```json ```."
                })

        return {"error": "Failed to parse document complying with schema within retries limit"}
