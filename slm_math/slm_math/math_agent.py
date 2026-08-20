import os
import re
import sys
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
        cleaned = problem.strip()
        # Remove common instruction prefixes
        cleaned = re.sub(r'^(?:solve|calculate|evaluate|find|simplify|compute)\s+(?:this\s+)?(?:math\s+)?(?:equation|problem|integral|derivative|expression)?\s*(?:step-by-step)?\s*[:\-\?]?\s*', '', cleaned, flags=re.IGNORECASE).strip()
        
        # Check for integration
        if "integrate" in cleaned.lower():
            if re.fullmatch(r"integrate\s*\(.+\)", cleaned, re.IGNORECASE):
                return cleaned
            match = re.search(r"integrate\s+(.+?)\s+from\s+(.+?)\s+to\s+(.+)", cleaned, re.IGNORECASE)
            if match:
                expr_str, var_start, var_end = match.group(1), match.group(2), match.group(3)
                return f"integrate({expr_str.strip()}, {var_start.strip()}, {var_end.strip()})"
                
        # Check for equation with =
        eq_match = re.search(r'([0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+\s*=\s*[0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+)', cleaned)
        if eq_match:
            return eq_match.group(1).strip()
            
        # Check for mathematical expression
        math_match = re.search(r'([0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+(?:\s*[\+\-\*\/\^]\s*[0-9a-zA-Z\s\+\-\*\/\^\(\)\.]+)+)', cleaned)
        if math_match:
            return math_match.group(1).strip()
            
        return cleaned

    def solve(self, problem_description: str, system_prompt: str = None, user_input: str = None) -> dict:
        if not problem_description:
            return {"success": False, "error": "Empty problem description"}

        eq_str = self._extract_equation(problem_description)
        result_val = None
        steps = [f"Identified mathematical formulation: `{eq_str}`"]

        if sp is not None:
            try:
                # 1. Definite Integration
                if eq_str.startswith("integrate"):
                    parts = eq_str[len("integrate("):-1].split(",")
                    if len(parts) == 3:
                        expr_s = parts[0].strip().replace("^", "**")
                        low_s, high_s = parts[1].strip(), parts[2].strip()
                        x = sp.Symbol('x')
                        expr = sp.sympify(expr_s)
                        val = sp.integrate(expr, (x, sp.sympify(low_s), sp.sympify(high_s)))
                        result_val = str(val)
                        steps.append(f"Set up definite integral $\\int_{{{low_s}}}^{{{high_s}}} ({sp.latex(expr)}) \\, dx$.")
                        steps.append(f"Found antiderivative and evaluated boundary values from {low_s} to {high_s}.")
                        steps.append(f"Computed exact symbolic result: `{result_val}`.")
                
                # 2. Algebraic Equation with Equality (=)
                elif "=" in eq_str:
                    lhs_str, rhs_str = eq_str.split("=", 1)
                    lhs_cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', lhs_str.strip()).replace("^", "**")
                    rhs_cleaned = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', rhs_str.strip()).replace("^", "**")
                    
                    x = sp.Symbol('x')
                    lhs_expr = sp.sympify(lhs_cleaned)
                    rhs_expr = sp.sympify(rhs_cleaned)
                    eq = sp.Eq(lhs_expr, rhs_expr)
                    
                    # Standard form
                    diff_expr = sp.simplify(lhs_expr - rhs_expr)
                    steps.append(f"Re-arranged into standard form: `{diff_expr} = 0`.")
                    
                    # Check polynomial degree
                    poly = diff_expr.as_poly(x) if hasattr(diff_expr, 'as_poly') else None
                    if poly and poly.degree() == 2:
                        coeffs = poly.all_coeffs()
                        if len(coeffs) == 3:
                            a, b, c = coeffs[0], coeffs[1], coeffs[2]
                            disc = b**2 - 4*a*c
                            steps.append(f"Identified quadratic coefficients: $a = {a}$, $b = {b}$, $c = {c}$.")
                            steps.append(f"Calculated discriminant $\\Delta = b^2 - 4ac = ({b})^2 - 4({a})({c}) = {disc}$.")
                            steps.append(f"Applied quadratic formula: $x = \\frac{{-b \\pm \\sqrt{{\\Delta}}}}{{2a}} = \\frac{{-({b}) \\pm \\sqrt{{{disc}}}}}{{2({a})}}$.")
                            
                    sols = sp.solve(eq, x)
                    if sols:
                        formatted_sols = ", ".join([f"x = {s}" for s in sols])
                        result_val = formatted_sols
                        steps.append(f"Determined exact roots: **{formatted_sols}**.")
                
                # 3. Arithmetic / Algebraic Expression Evaluation
                else:
                    cleaned_expr = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', eq_str.strip()).replace("^", "**")
                    val = sp.sympify(cleaned_expr)
                    if val.is_number:
                        result_val = str(val) if not val.is_Float else f"{float(val):.6g}"
                    else:
                        simplified = sp.simplify(val)
                        result_val = str(simplified)
                    steps.append(f"Simplified and evaluated symbolic expression.")
                    steps.append(f"Computed exact value: `{result_val}`.")
            except Exception as e:
                result_val = None

        # Fallback to pure python evaluation if SymPy failed
        if result_val is None:
            integral_match = re.fullmatch(
                r"integrate\(\s*([+-]?\d*\.?\d*)?\s*\*?\s*x(?:\^(\d+))?\s*,\s*([+-]?\d+(?:\.\d+)?)\s*,\s*([+-]?\d+(?:\.\d+)?)\s*\)",
                eq_str,
                re.IGNORECASE,
            )
            if integral_match:
                coefficient_text, exponent_text, low_text, high_text = integral_match.groups()
                coefficient = -1.0 if coefficient_text == "-" else float(coefficient_text or 1)
                exponent = int(exponent_text or 1)
                low, high = float(low_text), float(high_text)
                value = coefficient * (high ** (exponent + 1) - low ** (exponent + 1)) / (exponent + 1)
                result_val = str(int(value)) if value.is_integer() else f"{value:.12g}"
                steps.append("Evaluated the monomial antiderivative at the upper and lower bounds.")

        if result_val is None:
            try:
                safe_dict = {"__builtins__": None, "abs": abs, "min": min, "max": max}
                cleaned = re.sub(r"[^\d\.\+\-\*\/\(\)\s]", "", eq_str)
                if cleaned.strip():
                    result_val = str(eval(cleaned, safe_dict))
                    steps.append("Evaluated numeric arithmetic expression.")
            except Exception:
                pass

        if result_val is None:
            result_val = f"Manual resolution required for: {eq_str}"

        explanation = "\n".join([f"- {s}" for s in steps]) + f"\n\n**Final Answer**: **{result_val}**"
        return {
            "success": not result_val.startswith("Manual resolution required"),
            "problem": problem_description,
            "equation": eq_str,
            "result": result_val,
            "steps": steps,
            "explanation": explanation
        }
