import os
import json
import yaml
from typing import Tuple, Dict, Any, List
try:
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> Tuple[dict, str]:
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
        try:
            import main
            if hasattr(main, "get_shared_onnx_genai"):
                self.model, self.tokenizer = main.get_shared_onnx_genai()
                if self.model and self.tokenizer:
                    self.model_path = getattr(main, "MODEL_PATH", "shared_onnx")
                    return
        except Exception as e:
            print(f"[SLMTaskPlanner] Shared model note: {e}")

        if og is None:
            return
        base_models = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        candidates = [
            model_path,
            os.path.join(base_models, "qwen2.5-coder-3b-onnx"),
            os.path.join(base_models, "qwen3.5-0.8b-onnx"),
            os.path.join(base_models, "phi-3.5-mini-instruct-onnx", "cpu_and_mobile", "cpu-int4-awq-block-128-acc-level-4"),
            self.config.get("model", {}).get("path")
        ]
        resolved = None
        for c in candidates:
            if c and os.path.exists(c):
                resolved = c
                break
        if resolved:
            try:
                self.model_path = resolved
                self.model = og.Model(resolved)
                self.tokenizer = og.Tokenizer(self.model)
            except Exception as e:
                print(f"[SLMTaskPlanner] ONNX init note: {e}")

    def fetch_context_from_memory(self, session_id: str = None, query: str = None, history: list = None, memory_manager = None) -> dict:
        """
        Retrieves relevant contextual assets, active documents, user facts, and turn history from SLMMemoryManager.
        """
        context_data = {
            "has_context": False,
            "context_summary": "",
            "active_document": None,
            "facts": [],
            "recent_turns": []
        }
        if not session_id and not history and not query:
            return context_data

        try:
            mem_mgr = memory_manager
            if mem_mgr is None:
                try:
                    from slm_memory import SLMMemoryManager
                    mem_mgr = SLMMemoryManager()
                except Exception:
                    mem_mgr = None

            if mem_mgr:
                sid = session_id or "default_session"
                resolved = mem_mgr.resolve_context(sid, query or "", history or [])
                if resolved:
                    context_data["active_document"] = resolved.get("active_document")
                    context_data["recent_turns"] = resolved.get("recent_turns", [])
                    
                    ctx_parts = []
                    if resolved.get("has_active_document") and resolved.get("active_document"):
                        doc = resolved["active_document"]
                        doc_name = doc.get("name", "document")
                        ctx_parts.append(f"- Active Document in Memory: '{doc_name}'")

                    if hasattr(mem_mgr, "get_relevant_facts") and query:
                        facts = mem_mgr.get_relevant_facts(query)
                        if facts:
                            context_data["facts"] = facts
                            ctx_parts.append("- Relevant User Facts / Directives:\n  " + "\n  ".join([f"• {f}" for f in facts[:3]]))

                    if context_data["recent_turns"]:
                        turn_lines = []
                        for t in context_data["recent_turns"][-3:]:
                            u = str(t.get("user", ""))[:80]
                            a = str(t.get("assistant", ""))[:80]
                            turn_lines.append(f"User: {u} -> Assistant: {a}")
                        if turn_lines:
                            ctx_parts.append("- Recent Conversational History:\n  " + "\n  ".join(turn_lines))

                    if ctx_parts:
                        context_data["has_context"] = True
                        context_data["context_summary"] = "\n".join(ctx_parts)
        except Exception as e:
            print(f"[SLMTaskPlanner] Memory context retrieval note: {e}")

        return context_data

    def build_plan(self, goal: str, system_prompt: str = None, user_input: str = None, token_callback: callable = None, session_id: str = None, history: list = None, memory_manager = None, **kwargs) -> dict:
        """
        Decomposes a user goal into detailed milestones with target agent assignments,
        incorporating context retrieved from SLMMemoryManager.
        """
        if not goal or not str(goal).strip():
            return {"goal": "", "tasks": [], "total_steps": 0, "plan_markdown": "", "status": "error"}

        clean_goal = str(goal).strip()
        if "[Current Task]:" in clean_goal:
            clean_goal = clean_goal.split("[Current Task]:")[-1].strip()

        # Retrieve relevant memory context from SLMMemoryManager
        mem_ctx = self.fetch_context_from_memory(
            session_id=session_id or kwargs.get("session_id"),
            query=clean_goal,
            history=history or kwargs.get("history"),
            memory_manager=memory_manager or kwargs.get("memory_manager")
        )

        # Re-verify model availability
        if getattr(self, "model", None) is None or getattr(self, "tokenizer", None) is None:
            self._init_model()

        # 1. Primary: Use LLM Neural Engine to generate rich, tailored milestone roadmap
        if getattr(self, "model", None) is not None and getattr(self, "tokenizer", None) is not None and og is not None:
            sys_prompt = (
                "You are an expert Chief Technology Officer & Technical Project Architect.\n"
                "Understand the user's objective and any active memory context, then formulate a comprehensive, highly structured 4-phase milestone roadmap.\n"
                "Assign specialized SLM agents (e.g. SLMCodeInterpreter, SLMTextToSQL, SLMRag, SLMDataAnalyst, SLMSecurityAudit, SLMGitRepoManager, SLMCLIAgent).\n\n"
                "Use the following clean markdown format:\n\n"
                "### 📋 Strategic Roadmap: [Goal Name]\n\n"
                "#### 🔹 Phase 1: Architecture & Technical Discovery\n"
                "- Key Objectives & Deliverables\n"
                "- **Assigned Agent**: `SLMTaskPlanner` / `SLMSecurityAudit`\n\n"
                "#### 🔹 Phase 2: Core Engineering & Implementation\n"
                "- Key Objectives & Deliverables\n"
                "- **Assigned Agent**: `SLMCodeInterpreter` / `SLMTextToSQL`\n\n"
                "#### 🔹 Phase 3: Verification, Testing & Compliance\n"
                "- Security, Privacy & Performance validation\n"
                "- **Assigned Agent**: `SLMSecurityAudit` / `SLMDataAnalyst`\n\n"
                "#### 🔹 Phase 4: Production Deployment & Monitoring\n"
                "- Release, CI/CD and telemetry milestones\n"
                "- **Assigned Agent**: `SLMGitRepoManager` / `SLMCLIAgent`"
            )
            if mem_ctx["has_context"]:
                sys_prompt += f"\n\n[Active Session Memory Context]:\n{mem_ctx['context_summary']}"

            is_phi = "phi" in str(getattr(self, "model_path", "")).lower()
            if is_phi:
                full_prompt = (
                    f"<|system|>\n{sys_prompt}<|end|>\n"
                    f"<|user|>\nProject Goal: {clean_goal}<|end|>\n"
                    f"<|assistant|>\n"
                )
            else:
                full_prompt = (
                    f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
                    f"<|im_start|>user\nProject Goal: {clean_goal}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )
            try:
                input_tokens = self.tokenizer.encode(full_prompt)
                max_tokens = int(os.environ.get("SLM_TASK_PLANNER_MAX_TOKENS", 400))
                params = og.GeneratorParams(self.model)
                params.set_search_options(max_length=len(input_tokens) + max_tokens, temperature=0.2, repetition_penalty=1.18)
                generator = og.Generator(self.model, params)
                generator.append_tokens(input_tokens)
                
                out_tokens = []
                while not generator.is_done():
                    generator.generate_next_token()
                    new_toks = generator.get_next_tokens()
                    if len(new_toks) > 0:
                        tok_id = int(new_toks[0])
                        if tok_id in (151643, 151645, 248046, 248044, 248045, 32000, 32007) or tok_id >= 151936:
                            break
                        out_tokens.append(tok_id)
                        if token_callback:
                            try:
                                tok_str = self.tokenizer.decode([tok_id])
                                if tok_str and "<think>" not in tok_str and "</think>" not in tok_str:
                                    token_callback(tok_str)
                            except Exception:
                                pass
                        
                raw_plan = self.tokenizer.decode(out_tokens).strip()
                if "</think>" in raw_plan:
                    raw_plan = raw_plan.split("</think>")[-1].strip()
                elif "<think>" in raw_plan:
                    import re
                    raw_plan = re.sub(r'<think>.*?</think>', '', raw_plan, flags=re.DOTALL).strip()
                    raw_plan = re.sub(r'<think>.*', '', raw_plan, flags=re.DOTALL).strip()

                if raw_plan and len(raw_plan) >= 40:
                    return {
                        "goal": clean_goal,
                        "plan_markdown": raw_plan,
                        "retrieved_context": mem_ctx,
                        "status": "success"
                    }
            except Exception as e:
                print(f"[SLMTaskPlanner] Neural generation note: {e}")

        # 2. Dynamic Domain-Aware Milestone Decomposition Fallback
        q_lower = clean_goal.lower()
        if "pdf" in q_lower:
            tasks = [
                {"step": 1, "task": f"Extract and parse document structure for: {clean_goal}", "assigned_agent": "SLMPDFChat"}
            ]
        elif any(w in q_lower for w in ["app", "mobile", "ios", "android", "privacy"]):
            tasks = [
                {"step": 1, "task": "Define Privacy Architecture & Local Zero-Knowledge Data Models", "assigned_agent": "SLMSecurityAudit"},
                {"step": 2, "task": "Develop Core Application Logic & Offline Storage Engine", "assigned_agent": "SLMCodeInterpreter"},
                {"step": 3, "task": "Implement Secure Local Cryptography & Permission Handling", "assigned_agent": "SLMSecurityAudit"},
                {"step": 4, "task": "Execute Automated Unit Testing, Privacy Auditing & App Store Compliance", "assigned_agent": "SLMDataAnalyst"},
                {"step": 5, "task": "Automate Build Packaging, Release Tagging & Production Rollout", "assigned_agent": "SLMGitRepoManager"}
            ]
        elif any(w in q_lower for w in ["sql", "database", "data", "pipeline", "etl"]):
            tasks = [
                {"step": 1, "task": "Profile Data Schema, Relationships & Normalization Rules", "assigned_agent": "SLMTextToSQL"},
                {"step": 2, "task": "Construct High-Throughput Transformation & Indexing Pipelines", "assigned_agent": "SLMCodeInterpreter"},
                {"step": 3, "task": "Run Automated Data Integrity, Quality & Performance Benchmarks", "assigned_agent": "SLMDataAnalyst"},
                {"step": 4, "task": "Deploy Migration Scripts with Zero-Downtime Verification", "assigned_agent": "SLMDBMigrator"}
            ]
        else:
            tasks = [
                {"step": 1, "task": f"Analyze Technical Scope & Define Architecture for: {clean_goal}", "assigned_agent": "SLMTaskPlanner"},
                {"step": 2, "task": f"Implement Core Functional Modules & Business Logic for: {clean_goal}", "assigned_agent": "SLMCodeInterpreter"},
                {"step": 3, "task": "Conduct Verification, Quality Assurance & Delivery", "assigned_agent": "SLMSecurityAudit"}
            ]

        fallback_plan = (
            f"### 📋 Strategic Action Plan: {clean_goal}\n\n"
            + "\n".join([f"{task['step']}. **{task['task']}** ➔ `{task['assigned_agent']}`" for task in tasks])
        )
        if token_callback:
            try:
                token_callback(fallback_plan)
            except Exception:
                pass

        return {
            "goal": clean_goal,
            "tasks": tasks,
            "total_steps": len(tasks),
            "plan_markdown": fallback_plan,
            "status": "success"
        }

