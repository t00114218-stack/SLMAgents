import json
import os
import sys

try:
    # pyrefly: ignore [missing-import]
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> dict:
    """
    Searches for config.yaml in environment variables, CWD, parent dirs,
    and package installation directories.
    """
    try:
        import yaml
    except ImportError:
        return {}
        
    config_paths = [
        os.environ.get("SLM_ORCHESTRATOR_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[SLMOrchestrator] Warning: Failed to load config from {path}: {e}")
    return {}

class SLMOrchestrator:
    """
    A configurable semantic routing orchestrator powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    Routes user queries dynamically to custom lists of agents with robust JSON parsing constraints.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=2048, n_threads=4):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )
            
        # Resolve the ONNX model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMOrchestrator] Loading ONNX model from: {self.model_path}...")
        self.model = og.Model(self.model_path)
        self.tokenizer = og.Tokenizer(self.model)
            
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        """
        Locates or downloads the necessary ONNX model as defined in config.yaml.
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)

        # Check config.yaml
        config = load_config()
        model_config = config.get("models", {}).get("orchestrator")
        if not model_config:
            raise ValueError("models.orchestrator configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.orchestrator in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        
        # Check if genai_config.json exists recursively in config_path
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                return root
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        if not repo_id:
            raise ValueError(f"Model file not found at {config_path} and auto-download parameters (repo_id) are missing in config.yaml")
            
        print(f"[SLMOrchestrator] ONNX Model not found at configured path. Auto-downloading...")
        os.makedirs(config_path, exist_ok=True)
        
        from huggingface_hub import snapshot_download
        snapshot_download(
            repo_id=repo_id,
            local_dir=config_path,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        
        # Scan again to find actual directory containing genai_config.json
        for root, dirs, files in os.walk(config_path):
            if "genai_config.json" in files:
                print(f"[SLMOrchestrator] Resolved model directory containing genai_config.json: {root}")
                return root
                
        return config_path

    def route(self, agents: list, question: str) -> str:
        """
        Routes a user query to one of the custom agents based on their name and description.
        Returns the exact name of the selected agent.
        """
        if not agents:
            raise ValueError("The 'agents' list cannot be empty.")
            
        agent_names = [a["name"] for a in agents]
        
        # Dynamically map coding, rag, and general agents to construct few-shot examples using actual agent names
        coding_agent = agent_names[0]
        rag_agent = agent_names[0]
        general_agent = agent_names[0]
        
        for agent in agents:
            name_lower = agent["name"].lower()
            desc_lower = agent.get("description", "").lower()
            
            # Coding
            if any(w in name_lower or w in desc_lower for w in ["code", "write", "program", "develop"]):
                coding_agent = agent["name"]
            # RAG
            if any(w in name_lower or w in desc_lower for w in ["search", "retriev", "read", "scan", "find"]):
                rag_agent = agent["name"]
            # General
            if any(w in name_lower or w in desc_lower for w in ["general", "support", "chat", "explain"]):
                general_agent = agent["name"]
        
        # Build routing system prompt with JSON descriptions and dynamic few-shot examples
        agents_json = {a["name"]: {"description": a["description"]} for a in agents}
        
        system_prompt = (
            "You are a precise routing assistant. You are given a JSON containing agent details and descriptions, and a user query.\n"
            "Based on the agent descriptions, choose the most appropriate agent to handle the user's query.\n"
            "Output your decision as a valid JSON with the key 'selected_agent'. The value must be EXACTLY one of the available agent names.\n\n"
            f"Agent Details JSON:\n{json.dumps(agents_json, indent=2)}\n\n"
            "Examples:\n"
            "User: Write a python script called hello.py that prints hello world\n"
            f"Assistant: {{\"selected_agent\": \"{coding_agent}\"}}\n"
            "User: Search the codebase for Fibonacci\n"
            f"Assistant: {{\"selected_agent\": \"{rag_agent}\"}}\n"
            "User: What is git?\n"
            f"Assistant: {{\"selected_agent\": \"{general_agent}\"}}\n\n"
            "Output format:\n"
            "{\"selected_agent\": \"<agent_name>\"}"
        )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"User Query: {question}<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        
        input_tokens = self.tokenizer.encode(prompt)
        
        params = og.GeneratorParams(self.model)
        
        total_max_length = len(input_tokens) + 64
        search_options = {
            "max_length": total_max_length,
            "temperature": 0.0 # Greedy decoding for exact semantic routing matching
        }
        params.set_search_options(**search_options)
        
        generator = og.Generator(self.model, params)
        generator.append_tokens(input_tokens)
        
        output_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            new_tokens = generator.get_next_tokens()
            if len(new_tokens) > 0:
                output_tokens.append(int(new_tokens[0]))
                
        response_text = self.tokenizer.decode(output_tokens).strip()
        
        # Robust post-processing to extract and map routing decision
        # 1. Direct JSON parsing
        try:
            cleaned = response_text.replace("```json", "").replace("```", "").strip()
            data = json.loads(cleaned)
            selected = data.get("selected_agent")
            if selected:
                for name in agent_names:
                    if name.lower() == selected.lower():
                        return name
        except Exception:
            pass
            
        # 2. Regex fallback
        import re
        match = re.search(r'"selected_agent"\s*:\s*"([^"]+)"', response_text, re.IGNORECASE)
        if match:
            selected = match.group(1).strip()
            for name in agent_names:
                if name.lower() == selected.lower():
                    return name
                    
        # 3. Direct substring search fallback on agent names
        for name in agent_names:
            if name.lower() in response_text.lower():
                return name
                
        # 4. Semantic mapping for common short outputs (like coding, rag, general)
        selected_to_use = selected if selected else response_text
        selected_lower = selected_to_use.lower()
        
        # 4a. Coding category
        if any(w in selected_lower for w in ["coding", "code", "write", "develop", "program"]):
            for agent in agents:
                desc = agent.get("description", "").lower()
                name = agent["name"].lower()
                if any(w in desc or w in name for w in ["code", "write", "develop", "program", "coding", "technical"]):
                    return agent["name"]
                    
        # 4b. RAG/Search category
        if any(w in selected_lower for w in ["rag", "search", "retrieval", "find", "read"]):
            for agent in agents:
                desc = agent.get("description", "").lower()
                name = agent["name"].lower()
                if any(w in desc or w in name for w in ["search", "retriev", "read", "scan", "find", "info"]):
                    return agent["name"]
                    
        # 4c. General support category
        if any(w in selected_lower for w in ["general", "support", "chat", "explain"]):
            for agent in agents:
                desc = agent.get("description", "").lower()
                name = agent["name"].lower()
                if any(w in desc or w in name for w in ["general", "support", "chat", "explain", "greet"]):
                    return agent["name"]
                    
        # 5. Safe default fallback
        return agent_names[0]
