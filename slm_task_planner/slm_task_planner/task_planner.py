import os
import json
import yaml
try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_TASK_PLANNER_CONFIG"),
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

class SLMTaskPlanner:
    """
    Autonomous goal decomposition system powered by local ONNX SLM.
    Breaks complex tasks into prioritized action items, deliverables, and agent assignments.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.model = None
        self.tokenizer = None
        self._init_model(model_path)

    def _init_model(self, model_path=None):
        if og is None:
            return
        resolved = model_path or self.config.get("model", {}).get("path") or os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "qwen3.5-0.8b-onnx")
        if os.path.exists(resolved):
            try:
                self.model = og.Model(resolved)
                self.tokenizer = og.Tokenizer(self.model)
            except Exception as e:
                print(f"[SLMTaskPlanner] ONNX init note: {e}")

    def build_plan(self, goal: str, system_prompt: str = None, user_input: str = None) -> dict:
        """
        Decomposes a user goal into detailed milestones with target agent assignments.
        """
        if not goal or not str(goal).strip():
            return {"goal": "", "tasks": [], "total_steps": 0, "plan_markdown": ""}

        clean_goal = str(goal).strip()
        if "[Current Task]:" in clean_goal:
            clean_goal = clean_goal.split("[Current Task]:")[-1].strip()

        if "pdf" in clean_goal.lower() and any(word in clean_goal.lower() for word in ("extract", "parse", "stats")):
            tasks = [
                {"step": 1, "task": "Extract layout & tabular data from document", "assigned_agent": "SLMPDFChat / SLMDocumentParser"}
            ]
        else:
            tasks = [
                {"step": 1, "task": f"Clarify requirements and acceptance criteria for: {clean_goal}", "assigned_agent": "SLMTaskPlanner"},
                {"step": 2, "task": f"Implement the smallest working solution for: {clean_goal}", "assigned_agent": "SLMCodeInterpreter"},
                {"step": 3, "task": "Run verification and correct implementation failures", "assigned_agent": "SLMCodeInterpreter"},
            ]

        # Well-defined extraction goals do not need model generation.
        if len(tasks) == 1:
            return {"goal": clean_goal, "tasks": tasks, "total_steps": 1}

        # If ONNX model is available, use it to generate a rich milestone breakdown
        if self.model is not None and self.tokenizer is not None:
            sys_prompt = (
                "You are an expert Project Management & Technical Architecture Planner.\n"
                "Break down the user's project goal into a clear, structured multi-phase milestone roadmap.\n"
                "For each milestone phase, include:\n"
                "- Phase Name & Objective\n"
                "- Key Action Items & Technical Deliverables\n"
                "- Assigned Agent (e.g. SLMCodeInterpreter, SLMTextToSQL, SLMDataAnalyst, SLMSecurityAudit, SLMTaskPlanner)\n"
                "Provide a detailed, professional, and actionable plan."
            )
            full_prompt = (
                f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                f"<|im_start|>user\nProject Goal: {clean_goal}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + 1024, temperature=0.7, repetition_penalty=1.15)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                out_tokens = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_toks = generator.get_next_tokens()
                    if len(new_toks) > 0:
                        tok_id = int(new_toks[0])
                        if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007):
                            break
                        out_tokens.append(tok_id)
                        
                raw_plan = self.tokenizer.decode(out_tokens).strip()
                if "</think>" in raw_plan:
                    raw_plan = raw_plan.split("</think>")[-1].strip()
                elif "<think>" in raw_plan:
                    import re
                    raw_plan = re.sub(r'<think>.*?</think>', '', raw_plan, flags=re.DOTALL).strip()
                    raw_plan = re.sub(r'<think>.*', '', raw_plan, flags=re.DOTALL).strip()

                if not raw_plan or len(raw_plan) < 20:
                    raise ValueError("The planner generated an empty or incomplete plan")
                    
                return {
                    "goal": clean_goal,
                    "tasks": tasks,
                    "total_steps": len(tasks),
                    "plan_markdown": raw_plan,
                    "status": "success"
                }
            except Exception as e:
                print(f"[SLMTaskPlanner] Generation error: {e}")

        fallback_plan = "\n".join(
            [f"### 📋 Strategic Action Plan: {clean_goal}", ""]
            + [f"{task['step']}. **{task['task']}** ➔ `{task['assigned_agent']}`" for task in tasks]
        )
        return {
            "goal": clean_goal,
            "tasks": tasks,
            "total_steps": len(tasks),
            "plan_markdown": fallback_plan,
            "status": "success"
        }
