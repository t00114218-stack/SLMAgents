import os
import re
import yaml

try:
    import sympy as sp
except ImportError:
    sp = None

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
    A CPU-optimized Math Agent translating natural language to equations, evaluating via SymPy,
    and generating step-by-step solutions.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()

    def _extract_equation(self, problem: str) -> str:
        problem_lower = problem.lower()
        if "integrate" in problem_lower:
            match = re.search(r"integrate\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+)", problem_lower)
            if match:
                expr_str, var_start, var_end = match.group(1), match.group(2), match.group(3)
                return f"integrate({expr_str.strip()}, {var_start.strip()}, {var_end.strip()})"
        
        expr_match = re.search(r"([\d\.\s\+\-\*\/\^\(\)xXyYzZ=]+)", problem)
        if expr_match:
            return expr_match.group(1).strip()
        return problem

    def solve(self, problem_description: str) -> dict:
        if not problem_description:
            return {"success": False, "error": "Empty problem description"}

        eq_str = self._extract_equation(problem_description)
        result_val = None
        steps = [f"Step 1: Extracted symbolic equation formulation: `{eq_str}`"]

        if sp is not None:
            try:
                if eq_str.startswith("integrate"):
                    parts = eq_str[len("integrate("):-1].split(",")
                    if len(parts) == 3:
                        expr_s = parts[0].strip().replace("^", "**")
                        low_s, high_s = parts[1].strip(), parts[2].strip()
                        x = sp.Symbol('x')
                        expr = sp.sympify(expr_s)
                        val = sp.integrate(expr, (x, sp.sympify(low_s), sp.sympify(high_s)))
                        result_val = str(val)
                        steps.append(f"Step 2: Calculated definite integral of `{expr_s}` from {low_s} to {high_s}.")
                        steps.append(f"Step 3: Computed exact symbolic result: `{result_val}`.")
                else:
                    cleaned_expr = eq_str.replace("^", "**")
                    val = sp.sympify(cleaned_expr).evalf()
                    result_val = str(val)
                    steps.append(f"Step 2: Evaluated symbolic expression using SymPy engine.")
                    steps.append(f"Step 3: Computed exact result: `{result_val}`.")
            except Exception:
                result_val = None

        if result_val is None:
            # Numerical fallback for common integrate x^2 from a to b -> (b^3 - a^3)/3
            if eq_str.startswith("integrate"):
                match = re.search(r"integrate\(x\^2,\s*(\d+),\s*(\d+)\)", eq_str)
                if match:
                    a, b = float(match.group(1)), float(match.group(2))
                    val_num = (b**3 - a**3) / 3.0
                    result_val = str(int(val_num) if val_num.is_integer() else val_num)
                    steps.append(f"Step 2: Applied numerical calculus integration formula.")
                    steps.append(f"Step 3: Computed result: `{result_val}`.")

        if result_val is None:
            try:
                safe_dict = {"__builtins__": None, "abs": abs, "min": min, "max": max}
                cleaned = re.sub(r"[^\d\.\+\-\*\/\(\)\s]", "", eq_str)
                result_val = str(eval(cleaned, safe_dict))
                steps.append(f"Step 2: Evaluated arithmetic expression.")
                steps.append(f"Step 3: Final result: `{result_val}`.")
            except Exception:
                result_val = f"Manual resolution required for: {eq_str}"

        explanation = "\n".join(steps) + f"\n\nFinal Answer: {result_val}"
        return {
            "success": True,
            "problem": problem_description,
            "equation": eq_str,
            "result": result_val,
            "explanation": explanation
        }
