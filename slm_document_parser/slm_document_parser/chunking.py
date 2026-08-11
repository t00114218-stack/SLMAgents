import os
import re
import json
import onnxruntime_genai as og

class SLMChunker:
    """Encapsulates local LLM semantic graph-chunking and linkage resolution."""
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _extract_json(self, text: str) -> str:
        match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            return brace_match.group(1).strip()
        return text.strip()

    def segment_and_extract_chunks(self, full_markdown: str, source_name: str) -> list[dict]:
        """Uses Phi-3.5 ONNX to segment markdown text into structured semantic chunks with metadata."""
        schema_dict = {
            "chunks": [
                {
                    "text": "string (the semantic text content of this chunk, representing a complete, meaningful paragraph, minimum 15-20 words)",
                    "heading": "string (the active section heading, e.g., '1. Overview')",
                    "subheading": "string (the active subsection heading or empty string)",
                    "product": "string (product, entity, or service name referenced, or empty string)",
                    "key_terms": ["list of strings (keywords or topics mentioned)"]
                }
            ]
        }
        
        system_prompt = (
            "You are a local RAG Semantic Chunking Agent.\n"
            "Analyze the document Markdown and divide it into meaningful semantic chunks based on topic changes.\n"
            "CRITICAL: Each semantic chunk MUST be a meaningful, paragraph-like block of complete sentences. "
            "NEVER create chunks containing only a single word, a page number, or tiny fragmented header lines. "
            "Each chunk must consist of at least 15-20 words to ensure it provides rich context for RAG.\n"
            "For each chunk, extract the active section heading, subheading, any product or company name referenced, and key terms.\n"
            "IMPORTANT: Output your result as a single valid JSON block complying with the target schema structure.\n"
            "Do not include any explanations outside of the JSON block."
        )
        
        user_prompt = (
            f"Document Content to Chunk:\n{full_markdown}\n\n"
            f"Target JSON Schema structure to populate:\n{json.dumps(schema_dict, indent=2)}"
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
            params.set_search_options(max_length=len(input_tokens) + 1500, temperature=0.0)
            
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
                if isinstance(parsed, dict) and "chunks" in parsed:
                    chunks_list = parsed["chunks"]
                    formatted_chunks = []
                    for idx, c in enumerate(chunks_list):
                        text_val = c.get("text", "").strip()
                        # Clean and filter out tiny, one-word, or meaningless snippets (minimum 15 words)
                        if len(text_val.split()) >= 15:
                            formatted_chunks.append({
                                "text": text_val,
                                "metadata": {
                                    "source": os.path.basename(source_name),
                                    "heading": c.get("heading", "").strip(),
                                    "subheading": c.get("subheading", "").strip(),
                                    "product": c.get("product", "").strip(),
                                    "key_terms": c.get("key_terms", []),
                                    "format": os.path.splitext(source_name)[1][1:].lower()
                                }
                            })
                    return formatted_chunks
            except Exception as e:
                messages.append({"role": "assistant", "content": response_text})
                messages.append({
                    "role": "user",
                    "content": f"JSON decoding failed: {e}. Correct the formatting and return the complete updated JSON inside ```json ```."
                })
                
        print("[SLMDocumentParser] LLM Semantic Chunker failed. Falling back to paragraph chunker.")
        return self.fallback_semantic_chunker(full_markdown, source_name)

    def fallback_semantic_chunker(self, text: str, source_name: str) -> list[dict]:
        """Fallback chunker that splits text by paragraphs and extracts basic headers."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_h1 = ""
        current_h2 = ""
        
        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue
                
            if para.startswith("# "):
                current_h1 = para[2:].strip()
                continue
            elif para.startswith("## "):
                current_h2 = para[3:].strip()
                continue
            elif para.startswith("### "):
                current_h2 = para[4:].strip()
                continue
                
            product = ""
            words = para.split()
            for w in words:
                w_clean = re.sub(r'[^\w]', '', w)
                if w_clean.istitle() and len(w_clean) > 4:
                    product = w_clean
                    break
                    
            if len(para.split()) >= 15:
                chunks.append({
                    "text": para,
                    "metadata": {
                        "source": os.path.basename(source_name),
                        "heading": current_h1,
                        "subheading": current_h2,
                        "product": product,
                        "key_terms": [current_h1] if current_h1 else ["general"],
                        "format": os.path.splitext(source_name)[1][1:].lower()
                    }
                })
            
        return chunks

    def link_chunks(self, chunks: list[dict]) -> list[dict]:
        """Post-processes chunks to link related section siblings and product references."""
        for idx, chunk in enumerate(chunks):
            chunk["metadata"]["chunk_index"] = idx
            chunk["metadata"]["related_chunks"] = []
            
        for i, c1 in enumerate(chunks):
            meta1 = c1["metadata"]
            idx1 = meta1["chunk_index"]
            related = set()
            
            for j in range(len(chunks)):
                if i == j:
                    continue
                c2 = chunks[j]
                meta2 = c2["metadata"]
                idx2 = meta2["chunk_index"]
                
                if meta1.get("heading") and meta1.get("heading") == meta2.get("heading"):
                    related.add(idx2)
                if meta1.get("subheading") and meta1.get("subheading") == meta2.get("subheading"):
                    related.add(idx2)
                if meta1.get("product") and meta1.get("product").lower() == meta2.get("product", "").lower():
                    related.add(idx2)
                    
                kt1 = set(meta1.get("key_terms", []))
                kt2 = set(meta2.get("key_terms", []))
                shared_terms = kt1.intersection(kt2)
                if len(shared_terms) >= 2:
                    related.add(idx2)
                    
            meta1["related_chunks"] = sorted(list(related))
            
        return chunks
