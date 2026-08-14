import json
import os
import sys

try:
    # pyrefly: ignore [missing-import]
    import onnxruntime_genai as og
except ImportError:
    og = None

def load_config() -> tuple[dict, str]:
    """
    Searches for config.yaml in environment variables, CWD, parent dirs,
    and package installation directories.
    Returns a tuple of (config_dict, config_file_path).
    """
    try:
        import yaml
    except ImportError:
        return {}, ""
        
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
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception as e:
                print(f"[SLMOrchestrator] Warning: Failed to load config from {path}: {e}")
    return {}, ""

class SLMOrchestrator:
    """
    A configurable semantic routing orchestrator powered by a local Small Language Model (SLM)
    running via ONNX Runtime GenAI.
    Routes user queries dynamically to custom lists of agents with robust JSON parsing constraints.

    Configuration can be set via constructor arguments OR environment variables:
      SLM_ORCHESTRATOR_CACHE_DIR  — Override model download/cache directory
      SLM_ORCHESTRATOR_N_THREADS  — Number of CPU threads (default: 4)
      SLM_ORCHESTRATOR_N_CTX      — Context window size (default: 2048)
      SLM_ORCHESTRATOR_CONFIG     — Path to a custom config.yaml file
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=None, n_threads=None):
        if og is None:
            raise ImportError(
                "onnxruntime-genai is not installed. Please install it using: "
                "pip install onnxruntime-genai"
            )

        # Resolve parameters: constructor args > env vars > defaults
        n_threads = n_threads or int(os.environ.get("SLM_ORCHESTRATOR_N_THREADS", 4))
        n_ctx     = n_ctx     or int(os.environ.get("SLM_ORCHESTRATOR_N_CTX", 2048))
        cache_dir = cache_dir or os.environ.get("SLM_ORCHESTRATOR_CACHE_DIR")

        # Wire thread count to ONNX Runtime (must be set before model load)
        os.environ["OMP_NUM_THREADS"] = str(n_threads)
        os.environ["MKL_NUM_THREADS"] = str(n_threads)
            
        # Resolve the ONNX model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        self.n_ctx = n_ctx
        
        print(f"[SLMOrchestrator] Loading ONNX model from: {self.model_path} (threads={n_threads})...")
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
        config, config_file_path = load_config()
        model_config = config.get("models", {}).get("orchestrator")
        if not model_config:
            raise ValueError("models.orchestrator configuration is missing in config.yaml")
            
        config_path = model_config.get("path")
        if not config_path:
            raise ValueError("model path configuration is missing under models.orchestrator in config.yaml")
            
        config_path = os.path.expanduser(config_path)
        if not os.path.isabs(config_path) and config_file_path:
            config_path = os.path.abspath(os.path.join(os.path.dirname(config_file_path), config_path))
        
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

    def route(self, agents: list, question: str, tools: list = None, tool_executor: callable = None, max_iterations: int = 5, system_prompt: str = None, user_input: str = None) -> str:
        """
        Routes a user query to one of the custom agents based on their name and description.
        Supports tool execution (e.g., Vector DB lookups) to gather more context before routing.
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
        )
        
        if tools and tool_executor:
            system_prompt += (
                f"\nAvailable Tools:\n{json.dumps(tools, indent=2)}\n"
                "If you need more information to decide, you can use a tool by outputting a JSON object with 'tool_call' and 'args' keys. Example:\n"
                "{\"tool_call\": \"search_vector_db\", \"args\": {\"query\": \"something\"}}\n"
            )
            
        system_prompt += (
            "Output your final routing decision as a valid JSON with the key 'selected_agent'. The value must be EXACTLY one of the available agent names.\n\n"
            f"Agent Details JSON:\n{json.dumps(agents_json, indent=2)}\n\n"
            "Examples:\n"
            "User: Write a python script called hello.py that prints hello world\n"
            f"Assistant: {{\"selected_agent\": \"{coding_agent}\"}}\n"
            "User: Search the codebase for Fibonacci\n"
            f"Assistant: {{\"selected_agent\": \"{rag_agent}\"}}\n"
            "User: What is git?\n"
            f"Assistant: {{\"selected_agent\": \"{general_agent}\"}}\n\n"
            "Output format for final routing decision:\n"
            "{\"selected_agent\": \"<agent_name>\"}"
        )
        
        prompt = (
            "<|im_start|>system\n"
            f"{system_prompt}<|im_end|>\n"
            "<|im_start|>user\n"
            f"User Query: {question}<|im_end|>\n"
        )
        
        for iteration in range(max_iterations):
            current_prompt = prompt + "<|im_start|>assistant\n"
            input_tokens = self.tokenizer.encode(current_prompt)
            
            params = og.GeneratorParams(self.model)
            
            total_max_length = len(input_tokens) + 64  # routing JSON only needs ~20 tokens
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
                    token_id = int(new_tokens[0])
                    if token_id in (151643, 151645):
                        break
                    output_tokens.append(token_id)
                    
            response_text = self.tokenizer.decode(output_tokens).strip()
            
            is_tool_call = False
            if tools and tool_executor:
                try:
                    cleaned = response_text.replace("```json", "").replace("```", "").strip()
                    data = json.loads(cleaned)
                    if "tool_call" in data:
                        is_tool_call = True
                        tool_name = data["tool_call"]
                        args = data.get("args", {})
                        
                        try:
                            result = tool_executor(tool_name, args)
                        except Exception as e:
                            result = f"Error executing tool {tool_name}: {e}"
                            
                        prompt += (
                            "<|im_start|>assistant\n"
                            f"{response_text}<|im_end|>\n"
                            "<|im_start|>tool\n"
                            f"{result}<|im_end|>\n"
                        )
                except Exception:
                    pass
            
            if not is_tool_call:
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
                selected_to_use = selected if 'selected' in locals() and selected else response_text
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
                
        return agent_names[0]

    def execute(self, question: str, agents: list = None, agent_registry: dict = None, system_prompt: str = None, user_input: str = None, **kwargs) -> dict:
        """
        Full End-to-End Orchestrator Pipeline:
        1. Takes the user query and available agents (or uses default SLM agent suite).
        2. Semantically routes the query to the most capable agent.
        3. Dynamically invokes the target agent and gathers its output.
        4. Collects and formats the final synthesized response to present back to the user.
        """
        if not question or not str(question).strip():
            return {
                "query": "",
                "routed_agent": "None",
                "response": "Please provide a valid query or instruction for the orchestrator.",
                "status": "error"
            }

        query_str = str(question).strip()
        user_agents = agents or []

        # If no custom agents list is provided, default to the complete SLM Agent Ecosystem
        if not user_agents:
            user_agents = [
                {"name": "SLMCodeInterpreter", "description": "Python script generation, execution, mathematical computation, and debugging."},
                {"name": "SLMTextToSQL", "description": "Database query generation, SQL translation, schema queries, table joins, and aggregations."},
                {"name": "SLMRag", "description": "Document search, grounded retrieval, and factual knowledge Q&A."},
                {"name": "SLMSummarizer", "description": "Text condensation, article summarization, bullet-point highlights, and TL;DRs."},
                {"name": "SLMEmail", "description": "Professional email drafting, business messages, newsletters, and email communications."},
                {"name": "SLMTaskPlanner", "description": "Task breakdown, multi-step project planning, and action item scheduling."},
                {"name": "SLMMathAgent", "description": "Mathematical problem solving, algebra, calculus, and step-by-step calculations."},
                {"name": "SLMJsonCleaner", "description": "JSON repair, syntax error fixing, schema cleanup, and data formatting."},
                {"name": "SLMGitRepoManager", "description": "Git repository operations, commit creation, branches, and code management."},
                {"name": "SLMTranslationHub", "description": "Multilingual translation between English, Hindi, Tamil, Telugu, Spanish, French, and German."},
                {"name": "SLMGeneralAssistant", "description": "General conversational assistance, explanations, greetings, and definitions."}
            ]

        # 1. Route query to selected agent
        selected_agent = self.route(
            agents=user_agents,
            question=query_str,
            system_prompt=system_prompt,
            user_input=user_input
        )

        response_text = ""
        
        # 2. Check if user provided a custom callable in agent_registry
        if agent_registry and selected_agent in agent_registry and callable(agent_registry[selected_agent]):
            try:
                callable_fn = agent_registry[selected_agent]
                try:
                    res = callable_fn(query_str, system_prompt=system_prompt, user_input=user_input, **kwargs)
                except TypeError:
                    res = callable_fn(query_str)
                    
                if isinstance(res, dict) and "response" in res:
                    response_text = res["response"]
                else:
                    response_text = str(res)
            except Exception as e:
                response_text = f"Error executing custom agent {selected_agent}: {e}"
        else:
            # 3. Dynamic dispatch to SLM agent package
            agent_lower = selected_agent.lower().replace("_", "").replace("-", "")
            
            # Setup sys.path for workspace modules
            _curr_dir = os.path.dirname(os.path.abspath(__file__))
            _ws_dir = os.path.dirname(os.path.dirname(_curr_dir))
            if _ws_dir not in sys.path:
                sys.path.insert(0, _ws_dir)
                
            try:
                if "code" in agent_lower or "interpreter" in agent_lower:
                    from slm_code_interpreter import SLMCodeInterpreter
                    runner = SLMCodeInterpreter()
                    res = runner.generate_and_run(query_str)
                    response_text = res.get("output", "") or res.get("response", str(res))
                    
                elif "sql" in agent_lower or "database" in agent_lower:
                    from slm_text_to_sql import SLMTextToSQL
                    runner = SLMTextToSQL()
                    schema_hint = kwargs.get("schema", "table_data (id INT, name TEXT, value NUMERIC, date DATE)")
                    response_text = runner.generate_sql(schema=schema_hint, question=query_str)
                    
                elif "rag" in agent_lower or "retriev" in agent_lower:
                    from slm_rag import SLMRag
                    runner = SLMRag()
                    docs = kwargs.get("chunks") or kwargs.get("documents") or [query_str]
                    response_text = runner.query(question=query_str, chunks=docs)
                    
                elif "summariz" in agent_lower:
                    from slm_summarizer import SLMSummarizer
                    runner = SLMSummarizer()
                    response_text = runner.summarize(text=query_str)
                    
                elif "email" in agent_lower:
                    from slm_email import SLMEmailAssistant
                    runner = SLMEmailAssistant()
                    response_text = runner.generate_email(topic=query_str)
                    
                elif "task" in agent_lower or "planner" in agent_lower:
                    from slm_task_planner import SLMTaskPlanner
                    runner = SLMTaskPlanner()
                    response_text = runner.generate_plan(goal=query_str)
                    
                elif "math" in agent_lower:
                    from slm_math import SLMMathAgent
                    runner = SLMMathAgent()
                    response_text = runner.solve_math(query_str)
                    
                elif "json" in agent_lower or "clean" in agent_lower:
                    from slm_json_cleaner import SLMJsonCleaner
                    runner = SLMJsonCleaner()
                    response_text = runner.clean_json(query_str)
                    
                elif "git" in agent_lower or "repo" in agent_lower:
                    from slm_git_repo_manager import SLMGitRepoManager
                    runner = SLMGitRepoManager()
                    response_text = runner.analyze_command(query_str)
                    
                elif "translat" in agent_lower:
                    from slm_translation.translation_hub import SLMTranslationHub
                    runner = SLMTranslationHub()
                    target_lang = kwargs.get("target_lang", "hi")
                    response_text = runner.translate(query_str, source_lang="en", target_lang=target_lang)
                    
                else:
                    # General Direct Reasoning fallback using shared model
                    prompt = (
                        f"<|im_start|>system\n"
                        f"{system_prompt or 'You are a helpful and intelligent AI assistant powered by local SLM.'}<|im_end|>\n"
                        f"<|im_start|>user\n{query_str}<|im_end|>\n<|im_start|>assistant\n"
                    )
                    input_tokens = self.tokenizer.encode(prompt)
                    params = og.GeneratorParams(self.model)
                    params.set_search_options(max_length=len(input_tokens) + 256, temperature=0.7, top_p=0.9)
                    generator = og.Generator(self.model, params)
                    generator.append_tokens(input_tokens)
                    
                    tokens_out = []
                    while not generator.is_done():
                        generator.generate_next_token()
                        new_tokens = generator.get_next_tokens()
                        if len(new_tokens) > 0:
                            token_id = int(new_tokens[0])
                            if token_id in (151643, 151645):
                                break
                            tokens_out.append(token_id)
                    response_text = self.tokenizer.decode(tokens_out).strip()
            except Exception as e:
                response_text = f"Agent {selected_agent} processed query '{query_str}'. (Execution note: {e})"

        # 4. Format and present response back to user
        return {
            "query": query_str,
            "routed_agent": selected_agent,
            "response": response_text,
            "status": "success"
        }
