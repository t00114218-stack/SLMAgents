import os
import re
import sys
import yaml

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_TRANSLATION_CONFIG"),
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

INDIAN_LANGUAGES = {"hi", "ta", "te", "bn", "mr", "gu", "kn", "pa", "ml", "or", "as"}

class SLMTranslationHub:
    """
    A local CPU-optimized Translation Hub supporting dynamic neural translation for Indian & global language pairs.
    """
    def __init__(self):
        self.config, _ = load_config()
        self.active_pair = None
        self.loaded_model = None
        self.model = None
        self.tokenizer = None
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
        except Exception:
            pass

    def _clean_text(self, text: str) -> str:
        if "</think>" in text:
            text = text.split("</think>")[-1].strip()
        elif "<think>" in text:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
            text = re.sub(r'<think>.*', '', text, flags=re.DOTALL).strip()
        return text.strip()

    def translate(self, text: str, source_lang: str = "en", target_lang: str = "hi", system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs) -> str:
        """
        Translates text from source_lang to target_lang while preserving formatting syntax.
        """
        if not text or not text.strip():
            return ""

        # Neural translation via ONNX
        if self.model and self.tokenizer and og is not None:
            sys_prompt = system_prompt or (
                f"You are a professional translator fluent in all major world and Indian languages.\n"
                f"Translate the provided text faithfully and naturally from {source_lang.upper()} to {target_lang.upper()}.\n"
                f"Preserve all Markdown formatting, technical terms, numbers, and code blocks unchanged.\n"
                f"Do not think out loud or output any <think> tags. Return ONLY the final translated text directly."
            )
            full_prompt = (
                "<|im_start|>system\n"
                f"{sys_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                f"Text to translate to {target_lang.upper()}:\n{text}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                max_tokens = int(os.environ.get("SLM_TRANSLATION_MAX_TOKENS", 3000))
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.15)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)

                tokens = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_tokens = generator.get_next_tokens()
                    if len(new_tokens) > 0:
                        tid = int(new_tokens[0])
                        if tid in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        tok_str = self.tokenizer.decode(new_tokens)
                        tokens.append(tok_str)
                        if token_callback:
                            try:
                                token_callback(tok_str)
                            except Exception:
                                pass

                gen_text = self._clean_text("".join(tokens))
                if gen_text:
                    return gen_text
            except Exception as e:
                print(f"[SLMTranslationHub] Neural generation note: {e}")

        # Simple fallback
        return f"[{target_lang.upper()} Translation of '{text}']"
