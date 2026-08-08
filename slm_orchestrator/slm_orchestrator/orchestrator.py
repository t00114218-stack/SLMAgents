import json
import os
import sys
import math
from llama_cpp import Llama, LlamaGrammar

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
    A configurable semantic routing orchestrator powered by a local Small Language Model (SLM).
    Can route user queries dynamically to custom lists of agents with strict GBNF grammar enforcement.
    """
    def __init__(self, model_path=None, cache_dir=None, n_ctx=1024, n_threads=4):
        # Resolve the GGUF model path
        self.model_path = self._resolve_model_path(model_path, cache_dir)
        
        # Load the local Llama model
        print(f"[SLMOrchestrator] Loading model from: {self.model_path}...")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=True,
                embedding=True,
                verbose=False
            )
        except Exception as e:
            print(f"[SLMOrchestrator] Warning: Failed to load with use_mlock=True: {e}. Retrying without mlock...")
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=n_ctx,
                n_threads=n_threads,
                use_mlock=False,
                embedding=True,
                verbose=False
            )
            
    def _resolve_model_path(self, model_path=None, cache_dir=None) -> str:
        """
        Locates or downloads the necessary GGUF model as defined in config.yaml.
        Precedence:
        1. Explicitly provided `model_path`
        2. Configured path/download via config.yaml
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
        if os.path.exists(config_path):
            return config_path
            
        # Download if configured but not present
        repo_id = model_config.get("repo_id")
        filename = model_config.get("filename")
        if not repo_id or not filename:
            raise ValueError(f"Model file not found at {config_path} and auto-download parameters (repo_id, filename) are missing in config.yaml")
            
        print(f"[SLMOrchestrator] Model not found at configured path. Auto-downloading...")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=os.path.dirname(config_path)
        )
        if downloaded != config_path and os.path.exists(downloaded):
            os.rename(downloaded, config_path)
        return config_path

    def _build_gbnf_grammar(self, agent_names: list) -> LlamaGrammar:
        """
        Dynamically constructs a GBNF grammar that restricts the model's output
        to a valid JSON with key 'selected_agent' holding one of the exact agent names.
        """
        escaped_names = []
        for name in agent_names:
            escaped = name.replace('\\', '\\\\').replace('"', '\\"')
            escaped_names.append(f'"\\"{escaped}\\""')
            
        agent_rules = " | ".join(escaped_names)
        
        grammar_string = f'''
            root   ::= "{{" space "\\"selected_agent\\":" space agent space "}}"
            agent  ::= {agent_rules}
            space  ::= [ \\t\\n\\r]*
        '''
        return LlamaGrammar.from_string(grammar_string)

    def route(self, agents: list, question: str) -> str:
        """
        Routes a user query to one of the custom agents based on their name and description.
        Returns the exact name of the selected agent.
        """
        if not agents:
            raise ValueError("The 'agents' list cannot be empty.")
            
        agent_names = [a["name"] for a in agents]
        
        # Build dynamic GBNF grammar
        grammar = self._build_gbnf_grammar(agent_names)
        
        # Build routing system prompt with JSON descriptions and few-shot examples
        agents_json = {a["name"]: {"description": a["description"]} for a in agents}
        system_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n\n"
            "You are a routing agent. You are given a JSON containing agent details and descriptions, and a user query.\n"
            "Based on the agent descriptions, choose the most appropriate agent to handle the user's query.\n"
            "Output your decision as a valid JSON with the key 'selected_agent'. The value must be EXACTLY one of the available agent names.\n\n"
            f"Agent Details JSON:\n{json.dumps(agents_json, indent=2)}\n\n"
            "Examples:\n"
            "User: Write a python script called hello.py that prints hello world\n"
            "Assistant: {\"selected_agent\": \"coding\"}\n"
            "User: Search the codebase for Fibonacci\n"
            "Assistant: {\"selected_agent\": \"rag\"}\n"
            "User: What is git?\n"
            "Assistant: {\"selected_agent\": \"general\"}\n"
            "User: explain the benefits of separating a RAG agent from a Coding agent\n"
            "Assistant: {\"selected_agent\": \"general\"}\n\n"
            "Output format:\n"
            "{\"selected_agent\": \"<agent_name>\"}<|eot_id|>"
        )
        
        prompt = f"{system_prompt}<|start_header_id|>user<|end_header_id|>\nUser: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
        # CPU generation with strict grammar rule
        response = self.llm(
            prompt,
            max_tokens=64,
            temperature=0.0, # Greedy search for maximum routing accuracy
            grammar=grammar
        )
        
        raw_json = response["choices"][0]["text"].strip()
        try:
            plan = json.loads(raw_json)
            selected_agent = plan.get("selected_agent")
            if selected_agent not in agent_names:
                # Fallback to closest match if somehow json key parsing is weird
                raise ValueError("Returned agent name is not in list of options.")
            return selected_agent
        except Exception as e:
            print(f"[SLMOrchestrator Error] Failed to parse output '{raw_json}': {e}")
            # Fallback to first agent in list if something goes wrong
            return agent_names[0]
