import os
import sys
import yaml

# Resolve sibling package imports if in developer workspace environment
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_dir, "slm_document_parser"))
sys.path.insert(0, os.path.join(base_dir, "slm_rag"))

try:
    from slm_document_parser import SLMDocumentParser
except ImportError:
    SLMDocumentParser = None

try:
    from slm_rag import SLMRag
except ImportError:
    SLMRag = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_PDF_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMPDFChat:
    """
    A secure local CPU-optimized PDF Chat agent that reuses slm_document_parser for layout/table
    extraction and slm_rag for local vector QA.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        try:
            self.doc_parser = SLMDocumentParser(model_path=model_path) if SLMDocumentParser else None
        except Exception:
            self.doc_parser = None
        try:
            self.rag = SLMRag(model_path=model_path) if SLMRag else None
        except Exception:
            self.rag = None
        self.loaded_chunks = []
        self.pdf_path = None

    def load(self, pdf_path: str) -> dict:
        """
        Parses PDF layout, extracts paragraphs and tables via SLMDocumentParser,
        and indexes chunks for RAG queries.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self.pdf_path = pdf_path
        extracted_text = ""

        if self.doc_parser and hasattr(self.doc_parser, "parse_document"):
            doc_data = self.doc_parser.parse_document(pdf_path)
            extracted_text = doc_data.get("markdown", "") if isinstance(doc_data, dict) else str(doc_data)

        if not extracted_text:
            try:
                import fitz
                doc = fitz.open(pdf_path)
                pages = [page.get_text("text") or "" for page in doc]
                extracted_text = "\n".join(pages)
            except Exception:
                try:
                    import pdfplumber
                    with pdfplumber.open(pdf_path) as pdf:
                        pages = [page.extract_text() or "" for page in pdf.pages]
                        extracted_text = "\n".join(pages)
                except Exception as e:
                    return {
                        "success": False,
                        "file": pdf_path,
                        "total_chunks": 0,
                        "error": f"PDF text extraction failed: {e}"
                    }

        if not extracted_text.strip():
            return {
                "success": False,
                "file": pdf_path,
                "total_chunks": 0,
                "error": "The PDF contained no extractable text."
            }

        # Chunk text into ~500 char blocks
        raw_blocks = [block.strip() for block in extracted_text.split("\n\n") if block.strip()]
        self.loaded_chunks = raw_blocks or [extracted_text]

        return {
            "success": True,
            "file": pdf_path,
            "total_chunks": len(self.loaded_chunks)
        }

    def ask(self, question: str, system_prompt: str = None, user_input: str = None) -> str:
        """
        Queries the loaded PDF content using SLMRag.
        """
        if not self.loaded_chunks:
            return "No PDF document loaded. Please call `.load(pdf_path)` first."

        if self.rag:
            return self.rag.answer(
                chunks=self.loaded_chunks,
                question=question,
                instruction="Answer strictly based on the extracted PDF document content."
            )

        # Fallback keyword match if RAG model isn't active
        relevant = [c for c in self.loaded_chunks if any(word.lower() in c.lower() for word in question.split() if len(word) > 3)]
        context = " ".join(relevant[:2]) if relevant else self.loaded_chunks[0]
        return f"RAG model unavailable. Relevant extracted PDF context (not a generated answer): {context}"
