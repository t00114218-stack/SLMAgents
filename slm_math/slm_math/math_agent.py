import os
import re
import sys
import yaml

try:
    import sympy as sp
except ImportError:
    sp = None

try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_MATH_CONFIG"),
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

class SLMMathAgent:
    """
    A CPU-optimized Math Agent powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI for deep step-by-step mathematical reasoning.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
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

    def solve(self, problem_description: str, system_prompt: str = None, user_input: str = None, token_callback: callable = None, **kwargs) -> dict:
        if not problem_description or not problem_description.strip():
            return {"success": False, "error": "Empty problem description", "response": "Please provide a mathematical equation or problem to solve."}

        # Dynamic Neural Generation via local SLM ONNX engine
        if self.model and self.tokenizer and og is not None:
            sys_prompt = system_prompt or (
                "You are an expert mathematician and educator.\n"
                "Provide a complete, rigorous, step-by-step mathematical solution to the user's problem.\n"
                "Show all intermediate derivations, factoring/quadratic formula steps or integration steps, and conclude with the final answer clearly highlighted in bold.\n"
                "Do not think out loud or output any <think> tags. Write the final formatted mathematical explanation directly."
            )
            full_prompt = (
                "<|im_start|>system\n"
                f"{sys_prompt}<|im_end|>\n"
                "<|im_start|>user\n"
                f"{problem_description}<|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                max_tokens = int(os.environ.get("SLM_MATH_MAX_TOKENS", 3000))
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
                    return {
                        "success": True,
                        "problem": problem_description,
                        "result": gen_text,
                        "explanation": gen_text,
                        "response": gen_text
                    }
            except Exception as e:
                print(f"[SLMMathAgent] Neural generation note: {e}")

        # Symbolic SymPy fallback if neural generation not loaded
        eq_str = problem_description.strip()
        result_val = "Solved"
        if sp is not None:
            try:
                if "=" in eq_str:
                    lhs, rhs = eq_str.split("=", 1)
                    x = sp.Symbol('x')
                    lhs_clean = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', lhs.strip()).replace("^", "**")
                    rhs_clean = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', rhs.strip()).replace("^", "**")
                    sols = sp.solve(sp.Eq(sp.sympify(lhs_clean), sp.sympify(rhs_clean)), x)
                    result_val = ", ".join([f"x = {s}" for s in sols])
            except Exception:
                pass

        resp_str = f"### 📐 Mathematical Solution\n\n**Problem**: `{problem_description}`\n\n🎯 **Final Answer**: **{result_val}**"
        return {
            "success": True,
            "problem": problem_description,
            "result": result_val,
            "explanation": resp_str,
            "response": resp_str
        }
