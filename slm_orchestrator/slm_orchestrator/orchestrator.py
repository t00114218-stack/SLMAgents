import json
import os
import sys
import math
from llama_cpp import Llama, LlamaGrammar

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
        Locates or downloads the necessary 1B GGUF model.
        Precedence:
        1. Explicitly provided `model_path`
        2. Local workspace directory model file (`llama-3.2-1b-instruct-q4_k_m.gguf`)
        3. User cache directory (`~/.cache/slm_orchestrator/`)
        """
        if model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Provided model_path does not exist: {model_path}")
            return os.path.abspath(model_path)
            
        # Check current working directory
        cwd_model = os.path.join(os.getcwd(), "llama-3.2-1b-instruct-q4_k_m.gguf")
        if os.path.exists(cwd_model):
            return cwd_model
            
        # Check user cache directory
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.cache/slm_orchestrator")
        os.makedirs(cache_dir, exist_ok=True)
        
        cached_model = os.path.join(cache_dir, "llama-3.2-1b-instruct-q4_k_m.gguf")
        if not os.path.exists(cached_model):
            print(f"[SLMOrchestrator] Model not found locally. Auto-downloading to cache: {cached_model}...")
            from huggingface_hub import hf_hub_download
            
            # Download BARTOWSKI quantized Llama-3.2-1B model
            hf_hub_download(
                repo_id="bartowski/Llama-3.2-1B-Instruct-GGUF",
                filename="Llama-3.2-1B-Instruct-Q4_K_M.gguf",
                local_dir=cache_dir
            )
            
            # Ensure it is named exactly 'llama-3.2-1b-instruct-q4_k_m.gguf'
            downloaded_file = os.path.join(cache_dir, "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
            if os.path.exists(downloaded_file) and downloaded_file != cached_model:
                os.rename(downloaded_file, cached_model)
                
        return cached_model

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
        
        # Build routing system prompt
        agents_desc = "\n".join([f"- Name: {a['name']}\n  Description: {a['description']}" for a in agents])
        system_prompt = (
            "<|start_header_id|>system<|end_header_id|>\n"
            "You are a routing agent. You are given a list of available agents with their names and descriptions, and a user query.\n"
            "Based on the agent descriptions, select the single most appropriate agent name to handle the user's query.\n"
            "Output your decision as a valid JSON with the key 'selected_agent'. The value must be EXACTLY one of the available agent names.\n\n"
            f"Available Agents:\n{agents_desc}\n\n"
            "Output format:\n"
            "{\"selected_agent\": \"<agent_name>\"}<|eot_id|>"
        )
        
        prompt = f"{system_prompt}<|start_header_id|>user<|end_header_id|>\nQuery: {question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"
        
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
