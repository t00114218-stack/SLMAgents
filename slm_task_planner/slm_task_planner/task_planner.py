import os
import yaml

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
    Autonomous goal decomposition system. Breaks complex tasks into prioritized action items
    and assigns them to specialized local sub-agents.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()

    def build_plan(self, goal: str, system_prompt: str = None, user_input: str = None) -> dict:
        """
        Decomposes a user goal into sub-tasks with target agent assignments.
        """
        if not goal:
            return {"goal": "", "tasks": []}

        tasks = []
        goal_lower = goal.lower()

        if "extract" in goal_lower or "pdf" in goal_lower or "invoice" in goal_lower:
            tasks.append({
                "step": 1,
                "task": "Extract layout & tabular data from document",
                "assigned_agent": "SLMPDFChat / SLMDocumentParser"
            })
        if "calc" in goal_lower or "math" in goal_lower or "tax" in goal_lower or "sum" in goal_lower:
            tasks.append({
                "step": len(tasks) + 1,
                "task": "Perform exact math calculation and symbolic verification",
                "assigned_agent": "SLMMathAgent"
            })
        if "csv" in goal_lower or "save" in goal_lower or "excel" in goal_lower or "data" in goal_lower:
            tasks.append({
                "step": len(tasks) + 1,
                "task": "Save and summarize results in dataset format",
                "assigned_agent": "SLMDataAnalyst"
            })

        if not tasks:
            tasks = [
                {"step": 1, "task": f"Analyze requirement: {goal}", "assigned_agent": "SLMOrchestrator"},
                {"step": 2, "task": "Execute primary operation", "assigned_agent": "SLMCodeInterpreter"}
            ]

        return {
            "goal": goal,
            "tasks": tasks,
            "total_steps": len(tasks)
        }
