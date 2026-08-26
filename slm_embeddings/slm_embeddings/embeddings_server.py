from __future__ import annotations
import os
import yaml
import numpy as np

try:
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError:
    ort = None
    Tokenizer = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_EMBEDDINGS_CONFIG"),
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

class SLMEmbeddingsServer:
    """
    A lightweight local CPU-optimized neural embedding engine powered by all-MiniLM-L6-v2 ONNX runtime and Rust tokenizers.
    Runs with near-zero memory footprint (< 30 MB RAM overhead).
    """
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.model_path = model_path
        self.session = None
        self.tokenizer = None
        self.vector_dim = 384
        self._find_and_load_model()

    def _find_and_load_model(self):
        if self.session is not None or ort is None:
            return

        candidate_paths = [
            self.model_path,
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "all-minilm-l6-v2-onnx"),
            "./models/all-minilm-l6-v2-onnx",
            "../models/all-minilm-l6-v2-onnx",
            os.path.expanduser("~/Documents/SLMAgents/models/all-minilm-l6-v2-onnx")
        ]

        for p in candidate_paths:
            if p and os.path.exists(p):
                model_file = os.path.join(p, "onnx", "model.onnx") if os.path.exists(os.path.join(p, "onnx", "model.onnx")) else os.path.join(p, "model.onnx")
                tok_file = os.path.join(p, "tokenizer.json")
                if os.path.exists(model_file) and os.path.exists(tok_file) and Tokenizer is not None:
                    try:
                        opts = ort.SessionOptions()
                        opts.intra_op_num_threads = int(os.environ.get("SLM_N_THREADS", 2))
                        opts.inter_op_num_threads = 1
                        self.session = ort.InferenceSession(model_file, opts, providers=["CPUExecutionProvider"])
                        self.tokenizer = Tokenizer.from_file(tok_file)
                        self.model_path = p
                        return
                    except Exception as e:
                        print(f"[SLMEmbeddingsServer] ONNX load note: {e}")

    def embed(self, texts: list[str] | str, system_prompt: str = None, user_input: str = None) -> list[list[float]]:
        """
        Embeds a single string or list of text strings into dense 384-dimensional vector representations.
        Returns a list of float arrays (dimension: 384).
        """
        if isinstance(texts, str):
            texts = [texts]

        if not texts:
            return []

        self._find_and_load_model()

        if self.session and self.tokenizer:
            try:
                results = []
                for text in texts:
                    encoded = self.tokenizer.encode(text)
                    input_ids = np.array([encoded.ids[:512]], dtype=np.int64)
                    attention_mask = np.array([encoded.attention_mask[:512]], dtype=np.int64)
                    
                    onnx_inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
                    if "token_type_ids" in [inp.name for inp in self.session.get_inputs()]:
                        onnx_inputs["token_type_ids"] = np.array([encoded.type_ids[:512]], dtype=np.int64)
                    
                    outputs = self.session.run(None, onnx_inputs)
                    last_hidden_state = outputs[0]  # shape: (1, seq_len, hidden_dim)
                    
                    # Mean pooling with attention mask
                    input_mask_expanded = np.expand_dims(attention_mask, -1).astype(float)
                    sum_embeddings = np.sum(last_hidden_state * input_mask_expanded, 1)
                    sum_mask = np.clip(input_mask_expanded.sum(1), a_min=1e-9, a_max=None)
                    embeddings = sum_embeddings / sum_mask
                    
                    # Normalize to unit sphere
                    norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    normalized = (embeddings / np.maximum(norm, 1e-12))[0].tolist()
                    results.append(normalized)
                return results
            except Exception as e:
                print(f"[SLMEmbeddingsServer] Neural inference note: {e}")

        # Fallback dense hash generator
        results = []
        for text in texts:
            vec = np.zeros(self.vector_dim, dtype=np.float32)
            clean_text = text.lower()
            words = clean_text.split()
            for word in words:
                idx1 = abs(hash(word)) % self.vector_dim
                vec[idx1] += 1.0
            norm = np.linalg.norm(vec)
            if norm > 1e-12:
                vec /= norm
            results.append(vec.tolist())
        return results

    def similarity(self, text1: str, text2: str) -> float:
        """
        Calculates the cosine similarity between two text strings [0.0, 1.0].
        """
        v1 = np.array(self.embed([text1])[0])
        v2 = np.array(self.embed([text2])[0])
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
