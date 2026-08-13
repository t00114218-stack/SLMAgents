import os
import sys
import yaml
import json
import re
from pypdf import PdfReader
from docx import Document

try:
    from .chunking import SLMChunker
except ImportError:
    try:
        from chunking import SLMChunker  # type: ignore
    except ImportError:
        SLMChunker = None

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

    @property
    def chunker(self):
        if not hasattr(self, "_chunker"):
            model = getattr(self, "model", None)
            tokenizer = getattr(self, "tokenizer", None)
            self._chunker = SLMChunker(model, tokenizer) if SLMChunker else None
        return self._chunker

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
        """Extracts text by converting pages to images first (visual pipeline), falling back to native parsing."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        # Determine if we can run visual pipeline
        run_visual = False
        pdf_path = ""
        
        if ext == ".pdf":
            run_visual = True
            pdf_path = file_path
        elif ext in [".docx", ".doc", ".pptx", ".ppt"]:
            pdf_path = self._convert_office_to_pdf(file_path)
            if pdf_path:
                run_visual = True
                
        if run_visual and pdf_path:
            try:
                print(f"[SLMDocumentParser] Executing visual parser pipeline for: {file_path}...")
                reader = None
                total_pages = 0
                try:
                    import pypdfium2 as pdfium
                    doc = pdfium.PdfDocument(pdf_path)
                    total_pages = len(doc)
                    reader = PdfReader(pdf_path)
                except Exception as e:
                    print(f"[SLMDocumentParser] Failed to load PDF readers: {e}")
                
                if total_pages == 0:
                    raise ValueError("No pages found or PDF readers failed to load")
                
                page_markdowns = []
                for i in range(total_pages):
                    img_path = None
                    try:
                        digital_text = ""
                        if reader is not None and i < len(reader.pages):
                            try:
                                digital_text = reader.pages[i].extract_text() or ""
                            except Exception:
                                pass
                                
                        if len(digital_text.strip()) >= 10:
                            print(f"[SLMDocumentParser] Page {i+1}/{total_pages} contains digital text ({len(digital_text)} chars). Using hybrid text bypass.")
                            page_md = digital_text.strip()
                        else:
                            print(f"[SLMDocumentParser] Page {i+1}/{total_pages} is scanned/sparse. Rendering and running Florence-2 visual extraction...")
                            img_path = self._convert_page_to_image(pdf_path, i)
                            page_data = self._parse_page_visually(img_path)
                            page_md = self._assemble_and_correct_page_markdown(page_data, i + 1)
                            
                        page_markdowns.append(page_md)
                    finally:
                        if img_path and os.path.exists(img_path):
                            os.remove(img_path)
                            
                # Cleanup intermediate PDF if it was generated
                if pdf_path != file_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
                    
                return "\n\n".join(page_markdowns)
            except Exception as e:
                print(f"[SLMDocumentParser] Visual pipeline failed: {e}. Falling back to native parsing.")
                
        # --- NATIVE / TEXT FALLBACKS ---
        if ext == ".docx":
            try:
                doc = Document(file_path)
                text_blocks = []
                for p in doc.paragraphs:
                    if p.text.strip():
                        text_blocks.append(p.text)
                for table in doc.tables:
                    for row in table.rows:
                        row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                        if row_text:
                            text_blocks.append(row_text)
                return "\n".join(text_blocks)
            except Exception as e:
                print(f"[SLMDocumentParser] Native DOCX parser failed: {e}. Falling back to zip XML extraction.")
                return self._extract_docx_fallback(file_path)
                
        elif ext == ".doc":
            return self._extract_ole_text(file_path, "WordDocument")
            
        elif ext == ".pptx":
            try:
                from pptx import Presentation
                prs = Presentation(file_path)
                slides_text = []
                for slide in prs.slides:
                    slide_text = []
                    for shape in slide.shapes:
                        if hasattr(shape, "text") and shape.text.strip():
                            slide_text.append(shape.text.strip())
                    if slide_text:
                        slides_text.append("\n".join(slide_text))
                return "\n".join(slides_text)
            except Exception as e:
                print(f"[SLMDocumentParser] Native PPTX parser failed: {e}. Falling back to zip XML extraction.")
            return "\n".join([t for _, t in self._extract_pptx_fallback(file_path)])
            
        elif ext == ".ppt":
            return self._extract_ole_text(file_path, "PowerPoint Document")
            
        elif ext == ".pdf":
            # Native PDF text fallback
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
            
        else:
            # Fallback to plain text
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

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

    def _convert_page_to_image(self, pdf_path: str, page_idx: int) -> str:
        """Converts a specific PDF page index to a PNG image using pypdfium2."""
        try:
            import pypdfium2 as pdfium
        except ImportError:
            raise ImportError("pypdfium2 is required for PDF to image conversion.")
            
        doc = pdfium.PdfDocument(pdf_path)
        if page_idx < 0 or page_idx >= len(doc):
            return ""
            
        page = doc[page_idx]
        bitmap = page.render(scale=1.2)  # scale=1.2 is optimized for low-latency
        pil_image = bitmap.to_pil()
        
        import tempfile
        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, f"page_{page_idx+1}_{os.path.basename(pdf_path)}.png")
        pil_image.save(img_path)
        return img_path

    def _convert_office_to_pdf(self, file_path: str) -> str:
        """Converts doc/docx/ppt/pptx to pdf via LibreOffice CLI if available."""
        import subprocess
        import tempfile
        
        temp_dir = tempfile.gettempdir()
        try:
            soffice_cmd = "soffice"
            if sys.platform == "darwin":
                mac_soffice = "/Applications/LibreOffice.app/Contents/MacOS/soffice"
                if os.path.exists(mac_soffice):
                    soffice_cmd = mac_soffice
            
            cmd = [soffice_cmd, "--headless", "--convert-to", "pdf", "--outdir", temp_dir, file_path]
            subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=30)
            
            basename = os.path.splitext(os.path.basename(file_path))[0] + ".pdf"
            pdf_path = os.path.join(temp_dir, basename)
            if os.path.exists(pdf_path):
                return pdf_path
        except Exception as e:
            print(f"[SLMDocumentParser] LibreOffice conversion failed or not installed: {e}")
            
        return ""

    def _parse_page_visually(self, image_path: str) -> dict:
        """Parses a page image using Florence-2 to extract text, tables, and image captions."""
        vision = self._get_vision_parser()
        
        # 1. OCR the full page
        ocr_text = vision.parse_image(image_path, task="<OCR>")
        
        # 2. Detect objects (tables and figures)
        od_result_str = vision.parse_image(image_path, task="<OD>")
        
        import ast
        bboxes = []
        labels = []
        try:
            od_data = ast.literal_eval(od_result_str)
            if isinstance(od_data, dict):
                bboxes = od_data.get("bboxes", [])
                labels = od_data.get("labels", [])
        except Exception as e:
            print(f"[SLMDocumentParser] Failed to parse OD output: {e}")
            
        tables = []
        captions = []
        
        from PIL import Image
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        for box, label in zip(bboxes, labels):
            if len(box) == 4:
                ymin, xmin, ymax, xmax = box
                
                # Florence-2 OD outputs relative to 1000x1000
                if all(0 <= v <= 1000 for v in box) and (img_width != 1000 or img_height != 1000):
                    ymin = int(ymin * img_height / 1000)
                    ymax = int(ymax * img_height / 1000)
                    xmin = int(xmin * img_width / 1000)
                    xmax = int(xmax * img_width / 1000)
                
                ymin, ymax = max(0, int(ymin)), min(img_height, int(ymax))
                xmin, xmax = max(0, int(xmin)), min(img_width, int(xmax))
                
                if xmax > xmin and ymax > ymin:
                    cropped_img = img.crop((xmin, ymin, xmax, ymax))
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_f:
                        cropped_img.save(tmp_f.name)
                        tmp_path = tmp_f.name
                        
                    try:
                        if label == "table":
                            table_desc = vision.parse_image(tmp_path, task="<DETAILED_CAPTION>")
                            if table_desc.strip():
                                tables.append(table_desc.strip())
                        elif label in ["figure", "image", "chart", "diagram", "graph"]:
                            caption = vision.parse_image(tmp_path, task="<DETAILED_CAPTION>")
                            if caption.strip():
                                captions.append(caption.strip())
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)
                            
        return {
            "page_text": ocr_text,
            "tables": tables,
            "captions": captions
        }

    def _assemble_and_correct_page_markdown(self, page_data: dict, page_num: int) -> str:
        """Uses Phi-3.5 ONNX to structure and self-correct page OCR text, tables, and captions into clean Markdown."""
        page_text = page_data.get("page_text", "")
        tables_str = "\n\n".join([f"Table Area Description:\n{t}" for t in page_data.get("tables", [])])
        captions_str = "\n\n".join([f"Image/Chart Description: {c}" for c in page_data.get("captions", [])])
        
        system_prompt = (
            "You are a local Document Structuring Agent.\n"
            "Format the raw page OCR text, detected tables, and image/chart captions into a clean structured Markdown page.\n"
            "CRITICAL: Do NOT output tables as raw grid structures or pipe-delimited data. Instead, write a natural language text description explaining the table name, headers, columns, data fields, and values clearly.\n"
            "Integrate image/chart/diagram captions and descriptions close to their context.\n"
            "Maintain correct heading hierarchy (use #, ##, ###). Do not include any preamble or chatter, output ONLY Markdown."
        )
        
        user_prompt = (
            f"--- Page {page_num} ---\n"
            f"Raw OCR Text:\n{page_text}\n\n"
            f"{tables_str}\n\n"
            f"{captions_str}\n\n"
            "Output the structured Markdown for this page:"
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        max_retries = 3
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
                    
            validation_error = None
            lines = response_text.split("\n")
            for line in lines:
                if "|" in line:
                    if "---" in line and not any(c.isalnum() for c in line):
                        pass
            
            if not validation_error:
                return response_text.strip()
            else:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"Correction required: {validation_error}. Re-output the complete corrected Markdown."
                })
                
        return response_text.strip()

    def _segment_and_extract_chunks(self, full_markdown: str, source_name: str) -> list[dict]:
        """Uses Phi-3.5 ONNX to segment markdown text into structured semantic chunks with metadata."""
        if self.chunker:
            return self.chunker.segment_and_extract_chunks(full_markdown, source_name)
        return self._fallback_semantic_chunker(full_markdown, source_name)

    def _fallback_semantic_chunker(self, text: str, source_name: str) -> list[dict]:
        """Fallback chunker that splits text by paragraphs and extracts basic headers."""
        if self.chunker:
            return self.chunker.fallback_semantic_chunker(text, source_name)
        paragraphs = text.split("\n\n")
        chunks = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para.split()) >= 3:
                chunks.append({
                    "text": para,
                    "metadata": {
                        "source": os.path.basename(source_name),
                        "heading": "",
                        "subheading": "",
                        "product": "",
                        "key_terms": [],
                        "format": os.path.splitext(source_name)[1][1:].lower()
                    }
                })
        return chunks

    def _link_chunks(self, chunks: list[dict]) -> list[dict]:
        """Post-processes chunks to link related section siblings and product references."""
        if self.chunker:
            return self.chunker.link_chunks(chunks)
        return chunks

    def parse_and_chunk_stream(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
        """Visual document parser that yields chunks page-by-page as they are processed."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Document file not found: {file_path}")
            
        ext = os.path.splitext(file_path)[1].lower()
        
        run_visual = False
        pdf_path = ""
        if ext == ".pdf":
            run_visual = True
            pdf_path = file_path
        elif ext in [".docx", ".doc", ".pptx", ".ppt"]:
            pdf_path = self._convert_office_to_pdf(file_path)
            if pdf_path:
                run_visual = True
                
        all_chunks = []
        
        if run_visual and pdf_path:
            try:
                print(f"[SLMDocumentParser] Streaming visual parser pipeline for: {file_path}...")
                reader = None
                total_pages = 0
                try:
                    import pypdfium2 as pdfium
                    doc = pdfium.PdfDocument(pdf_path)
                    total_pages = len(doc)
                    reader = PdfReader(pdf_path)
                except Exception as e:
                    print(f"[SLMDocumentParser] Failed to load PDF readers: {e}")
                
                if total_pages == 0:
                    raise ValueError("No pages found or PDF readers failed to load")
                
                for i in range(total_pages):
                    img_path = None
                    try:
                        digital_text = ""
                        if reader is not None and i < len(reader.pages):
                            try:
                                digital_text = reader.pages[i].extract_text() or ""
                            except Exception:
                                pass
                                
                        if len(digital_text.strip()) >= 10:
                            print(f"[SLMDocumentParser] Page {i+1}/{total_pages} contains digital text ({len(digital_text)} chars). Using hybrid text bypass (low latency).")
                            page_md = digital_text.strip()
                        else:
                            print(f"[SLMDocumentParser] Page {i+1}/{total_pages} is scanned/sparse. Rendering and running Florence-2 visual extraction...")
                            img_path = self._convert_page_to_image(pdf_path, i)
                            page_data = self._parse_page_visually(img_path)
                            page_md = self._assemble_and_correct_page_markdown(page_data, i + 1)
                        
                        page_chunks = self._segment_and_extract_chunks(page_md, file_path)
                        
                        for c in page_chunks:
                            c["metadata"]["page_number"] = i + 1
                            if ext in [".pptx", ".ppt"]:
                                c["metadata"]["slide_number"] = i + 1
                                
                        start_new_idx = len(all_chunks)
                        all_chunks.extend(page_chunks)
                        self._link_chunks(all_chunks)
                        
                        for idx in range(start_new_idx, len(all_chunks)):
                            yield all_chunks[idx]
                            
                    finally:
                        if img_path and os.path.exists(img_path):
                            os.remove(img_path)
                            
                if pdf_path != file_path and os.path.exists(pdf_path):
                    os.remove(pdf_path)
                return
            except Exception as e:
                print(f"[SLMDocumentParser] Streaming visual pipeline failed: {e}. Falling back to native stream.")
                
        full_text = self.extract_text(file_path)
        semantic_chunks = self._segment_and_extract_chunks(full_text, file_path)
        all_chunks.extend(semantic_chunks)
        self._link_chunks(all_chunks)
        for chunk in all_chunks:
            yield chunk

    def chunk_document(self, file_path: str, chunk_size: int = 500, chunk_overlap: int = 50) -> list[dict]:
        """Runs the streaming parser and returns the complete list of linked semantic chunks."""
        return list(self.parse_and_chunk_stream(file_path, chunk_size, chunk_overlap))

    def export_chunks_to_excel(self, chunks: list[dict], output_path: str, append: bool = False) -> None:
        """Exports a list of chunk dicts to an Excel (.xlsx) file using openpyxl."""
        try:
            import openpyxl
        except ImportError:
            raise ImportError(
                "openpyxl is required to export to Excel. Please install it using:\n"
                "pip install openpyxl"
            )
            
        if append and os.path.exists(output_path):
            try:
                wb = openpyxl.load_workbook(output_path)
                ws = wb.active
                
                max_idx = -1
                for r in range(2, ws.max_row + 1):
                    val = ws.cell(row=r, column=1).value
                    if val is not None:
                        try:
                            max_idx = max(max_idx, int(val))
                        except ValueError:
                            pass
                start_idx = max_idx + 1
            except Exception as e:
                print(f"[SLMDocumentParser] Failed to load existing Excel file: {e}. Creating new sheet.")
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.title = "RAG Chunks"
                ws.append(["Chunk Index", "Source File", "Heading", "Subheading", "Product", "Related Chunks", "Text"])
                start_idx = 0
        else:
            wb = openpyxl.Workbook()
            ws = wb.active
            ws.title = "RAG Chunks"
            ws.append(["Chunk Index", "Source File", "Heading", "Subheading", "Product", "Related Chunks", "Text"])
            start_idx = 0
            
        current_idx = start_idx
        for chunk in chunks:
            metadata = chunk.get("metadata", {})
            related_str = ",".join(map(str, metadata.get("related_chunks", [])))
            row = [
                current_idx,
                metadata.get("source", ""),
                metadata.get("heading", ""),
                metadata.get("subheading", ""),
                metadata.get("product", ""),
                related_str,
                chunk.get("text", "")
            ]
            ws.append(row)
            current_idx += 1
            
        wb.save(output_path)
        print(f"[SLMDocumentParser] Exported {len(chunks)} chunks to Excel at: {output_path}")

    def _extract_docx_fallback(self, file_path: str) -> str:
        import zipfile
        import re
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                content = z.read('word/document.xml').decode('utf-8', errors='ignore')
                text_segments = re.findall(r'<w:t[^>]*>(.*?)</w:t>', content)
                text = "\n".join(text_segments)
                for k, v in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")]:
                    text = text.replace(k, v)
                return text
        except Exception as e:
            print(f"[SLMDocumentParser] Fallback DOCX extraction failed: {e}")
            return ""

    def _extract_pptx_fallback(self, file_path: str) -> list[tuple[int, str]]:
        import zipfile
        import re
        slides = []
        try:
            with zipfile.ZipFile(file_path, 'r') as z:
                slide_files = [f for f in z.namelist() if f.startswith('ppt/slides/slide') and f.endswith('.xml')]
                def extract_slide_num(name):
                    match = re.search(r'slide(\d+)\.xml', name)
                    return int(match.group(1)) if match else 0
                slide_files.sort(key=extract_slide_num)
                
                for file_name in slide_files:
                    slide_num = extract_slide_num(file_name)
                    content = z.read(file_name).decode('utf-8', errors='ignore')
                    text_segments = re.findall(r'<a:t[^>]*>(.*?)</a:t>', content)
                    text = " ".join(text_segments).strip()
                    if text:
                        for k, v in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'")]:
                            text = text.replace(k, v)
                        slides.append((slide_num, text))
        except Exception as e:
            print(f"[SLMDocumentParser] Fallback PPTX extraction failed: {e}")
        return slides

    def _extract_ole_text(self, file_path: str, stream_name: str) -> str:
        """Extracts text from OLE streams in legacy doc/ppt formats, falling back to binary string extraction."""
        try:
            import olefile
            if olefile.isOleFile(file_path):
                ole = olefile.OleFileIO(file_path)
                if ole.exists(stream_name):
                    stream = ole.openstream(stream_name)
                    data = stream.read()
                    return self._extract_strings_from_bytes(data)
        except Exception as e:
            print(f"[SLMDocumentParser] OLE stream extraction failed: {e}")
            
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            return self._extract_strings_from_bytes(data)
        except Exception as e:
            print(f"[SLMDocumentParser] OLE binary fallback failed: {e}")
            return ""

    def _extract_strings_from_bytes(self, data: bytes, min_len: int = 4) -> str:
        """Extracts readable ASCII/UTF-16 strings from binary stream payloads."""
        ascii_pattern = re.compile(rb'[\x20-\x7E\x0A\x0D\x09]{' + str(min_len).encode() + rb',}')
        ascii_matches = ascii_pattern.findall(data)
        
        utf16_pattern = re.compile(rb'(?:[\x20-\x7E\x0A\x0D\x09]\x00){' + str(min_len).encode() + rb',}')
        utf16_matches = utf16_pattern.findall(data)
        
        decoded = []
        for m in ascii_matches:
            try:
                decoded.append(m.decode('ascii').strip())
            except Exception:
                pass
        for m in utf16_matches:
            try:
                decoded.append(m.decode('utf-16le').strip())
            except Exception:
                pass
                
        return "\n".join([s for s in decoded if len(s) > 3])

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
        return text.strip()

    def parse(self, file_path: str, schema_dict: dict, max_retries: int = 3, system_prompt: str = None, user_input: str = None) -> dict:
        """Parses a document file into a structured JSON dict matching the schema_dict."""
        raw_text = self.extract_text(file_path)
        
        system_prompt = (
            "You are a local Document Parser agent.\n"
            "Analyze the document text and extract the data to populate a structured JSON block matching the target schema. "
            "IMPORTANT: Output actual extracted data. Never copy schema type descriptors (such as 'string', 'integer', 'boolean') or templates. "
            "Return the final completed JSON inside a ```json ... ``` code block. Never output explanations outside of the code block."
        )

        user_prompt = (
            f"Document Text:\n{raw_text[:8000]}\n\n"
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
                    "content": f"JSON parsing failed with error: {e}. Correct the JSON format and return the complete updated block inside ```json ```."
                })

        return {"error": "Failed to parse document complying with schema within retries limit"}
