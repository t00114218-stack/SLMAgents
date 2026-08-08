import json
import asyncio
import os
import sys
import re
import math
import random

# Auto-inject virtual environment's site-packages to sys.path if running under system python
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'lib', 'python3.9', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

from llama_cpp import Llama, LlamaGrammar

# 1. Initialize the 1B/3B Model entirely in RAM/CPU with embedding=True
print("[System] Loading orchestrator into CPU memory with embedding=True...")
try:
    llm = Llama(
        model_path="./qwen2.5-1.5b-instruct-q4_k_m.gguf", 
        n_ctx=1024,       # Context size
        n_threads=4,      # Physical CPU cores
        use_mlock=True,   # Lock RAM
        embedding=True,   # Enable embedding generation
        verbose=False
    )
except Exception as e:
    print(f"[Warning] Failed to load with use_mlock=True: {e}. Retrying without mlock...")
    llm = Llama(
        model_path="./qwen2.5-1.5b-instruct-q4_k_m.gguf", 
        n_ctx=1024,
        n_threads=4,
        use_mlock=False,
        embedding=True,
        verbose=False
    )

# 2. GBFN Grammar Guardrails for Agent Decision
json_grammar = LlamaGrammar.from_string(r'''
    root   ::= "{" space "\"selected_agent\":" space agent space "}"
    agent  ::= "\"rag\"" | "\"coding\"" | "\"general\""
    space  ::= [ \t\n\r]*
''')

# --- Embedding & Similarity Helpers ---
def get_embedding(text: str) -> list:
    """Generates a normalized 2048-dimensional embedding vector for a string using the local SLM."""
    try:
        res = llm.create_embedding(text)
        emb = res['data'][0]['embedding']
        # If it returns token-level embeddings (list of lists), perform mean pooling
        if isinstance(emb[0], list):
            num_tokens = len(emb)
            dim = len(emb[0])
            vector = [sum(emb[t][i] for t in range(num_tokens)) / num_tokens for i in range(dim)]
        else:
            vector = emb
        # Normalize vector for cosine similarity
        norm = math.sqrt(sum(x*x for x in vector))
        if norm > 0:
            vector = [x / norm for x in vector]
        return vector
    except Exception as e:
        print(f"[Embedding Error] {e}")
        return [0.0] * 2048

def cosine_similarity(v1: list, v2: list) -> float:
    """Computes cosine similarity between two normalized vectors."""
    return sum(x*y for x, y in zip(v1, v2))

# --- Workspace File Crawler ---
def scan_workspace() -> list:
    """Recursively scans the workspace for text/code files."""
    ignore_dirs = {
        "venv", ".git", "__pycache__", ".gemini", "node_modules", 
        ".vscode", ".idea", "build", "dist", ".cache"
    }
    ignore_extensions = {
        ".gguf", ".pyc", ".png", ".jpg", ".jpeg", ".gif", 
        ".zip", ".tar", ".gz", ".db", ".sqlite"
    }
    results = []
    workspace_path = os.getcwd()
    
    for root, dirs, files in os.walk(workspace_path):
        dirs[:] = [d for d in dirs if d not in ignore_dirs and not d.startswith(".")]
        
        for file in files:
            if file.startswith("."):
                continue
            _, ext = os.path.splitext(file)
            if ext.lower() in ignore_extensions:
                continue
                
            full_path = os.path.join(root, file)
            rel_path = os.path.relpath(full_path, workspace_path)
            results.append((rel_path, ""))
                
    return results

# --- Semantic RAG Agent ---
async def run_rag_agent(query: str) -> str:
    print(f"[RAG Agent] Performing local semantic vector search over the codebase...")
    await asyncio.sleep(0.5)
    
    files = scan_workspace()
    if not files:
        return "RAG: The workspace is empty."
        
    query_vector = get_embedding(query)
    all_chunks = []
    
    # Read files and segment into overlapping chunks of 5 lines
    for rel_path, _ in files:
        try:
            with open(rel_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            lines = content.splitlines()
            chunk_size = 5
            overlap = 2
            i = 0
            while i < len(lines):
                chunk_lines = lines[i:i+chunk_size]
                chunk_text = "\n".join(chunk_lines)
                if chunk_text.strip():
                    all_chunks.append({
                        "file": rel_path,
                        "start_line": i + 1,
                        "text": chunk_text
                    })
                i += (chunk_size - overlap)
        except Exception:
            pass
            
    if not all_chunks:
        return "RAG: No searchable text content in the workspace."
        
    # Calculate similarity score for each chunk
    scored_chunks = []
    for chunk in all_chunks:
        chunk_vector = get_embedding(chunk["text"])
        sim = cosine_similarity(query_vector, chunk_vector)
        scored_chunks.append((sim, chunk))
        
    # Sort chunks by similarity score descending
    scored_chunks.sort(key=lambda x: x[0], reverse=True)
    
    # Retrieve top matches
    top_matches = scored_chunks[:3]
    response_parts = [f"RAG: Top semantic matches in workspace for query '{query}':"]
    
    for idx, (sim, chunk) in enumerate(top_matches):
        if sim < 0.15: # Similarity threshold
            continue
        response_parts.append(f"\nMatch #{idx+1} (Similarity: {sim:.2f}) - File: {chunk['file']} (Line {chunk['start_line']}):")
        indented = "\n".join([f"  {l}" for l in chunk["text"].splitlines()])
        response_parts.append(indented)
        
    if len(response_parts) == 1:
        return f"RAG: No semantically relevant code chunks found for query '{query}'."
        
    return "\n".join(response_parts)

# --- Coding Agent ---
async def run_coding_agent(payload: str) -> str:
    print(f"[Coding Agent] Analyzing requested file operations...")
    
    system_prompt = (
        "<|im_start|>system\n"
        "You are an expert workspace developer. Based on the user's instructions, you must choose the action and file.\n"
        "Actions supported:\n"
        "- 'write': Create a new file or overwrite file contents with new code.\n"
        "- 'read': Read and output the existing code from a file.\n"
        "- 'edit': Modify an existing file. If you use 'edit', output the complete file contents including modifications.\n\n"
        "Respond EXACTLY in this format, with no markdown code blocks, explanations, or greeting lines:\n"
        "TARGET_FILE: <path_to_file>\n"
        "ACTION: <write / read / edit>\n"
        "CONTENT:\n"
        "<the contents to write or edit><|im_end|>"
    )
    
    prompt = f"{system_prompt}\n<|im_start|>user\nInstructions: {payload}<|im_end|>\n<|im_start|>assistant\n"
    
    # Run model generation (with temperature=0.1 to ensure structured formatting)
    response = llm(
        prompt,
        max_tokens=768,
        temperature=0.1
    )
    
    raw_response = response["choices"][0]["text"].strip()
    
    # Parsing headers from output
    lines = raw_response.splitlines()
    target_file = None
    action = None
    content_lines = []
    in_content = False
    
    for line in lines:
        if line.startswith("TARGET_FILE:"):
            target_file = line.replace("TARGET_FILE:", "").strip()
        elif line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip().lower()
        elif line.startswith("CONTENT:"):
            in_content = True
            rem = line.replace("CONTENT:", "", 1).strip()
            if rem:
                content_lines.append(rem)
        elif in_content:
            content_lines.append(line)
            
    content = "\n".join(content_lines).strip()
    
    # Parsing fallbacks
    if not target_file:
        file_match = re.search(r"TARGET_FILE:\s*(\S+)", raw_response)
        if file_match:
            target_file = file_match.group(1)
        else:
            fn_match = re.search(r"([\w-]+\.\w+)", payload)
            target_file = fn_match.group(1) if fn_match else "generated_code.py"
            
    if not action:
        act_match = re.search(r"ACTION:\s*(\w+)", raw_response)
        if act_match:
            action = act_match.group(1).lower()
        else:
            action = "read" if ("read" in payload.lower() or "show" in payload.lower()) else "write"
            
    # For security, keep operations within current workspace folder
    target_file = os.path.basename(target_file)
    print(f"[Coding Agent] Action Target -> {action.upper()} on file '{target_file}' (non-deterministic completion)")
    
    if action == "read":
        if os.path.exists(target_file):
            try:
                with open(target_file, "r", encoding="utf-8") as f:
                    content = f.read()
                return f"[Coding Agent] Successfully read file '{target_file}':\n\n{content}"
            except Exception as e:
                return f"[Coding Agent] Failed to read '{target_file}': {e}"
        else:
            return f"[Coding Agent] File '{target_file}' does not exist in workspace."
            
    elif action in ("write", "edit"):
        if not content:
            code_block = re.search(r"```[a-zA-Z]*\n(.*?)```", raw_response, re.DOTALL)
            if code_block:
                content = code_block.group(1).strip()
            else:
                content = "\n".join([l for l in lines if not l.startswith("TARGET_FILE") and not l.startswith("ACTION") and not l.startswith("CONTENT")]).strip()
                
        if not content:
            print(f"[Debug Error Output] Raw Coding Agent response was:\n{raw_response}")
            return f"[Coding Agent] Aborted. No content was generated to write to '{target_file}'."
            
        try:
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(content)
            return f"[Coding Agent] Successfully completed {action} on '{target_file}'. File Content:\n\n```\n{content}\n```"
        except Exception as e:
            return f"[Coding Agent] Failed to write '{target_file}': {e}"
            
    return f"[Coding Agent] Action '{action}' not recognized."

# --- General Agent ---
async def run_general_agent(payload: str) -> str:
    print(f"[General Agent] Handling general query...")
    system_prompt = (
        "<|im_start|>system\n"
        "You are a helpful software design assistant. Provide direct, informative and concise answers.<|im_end|>"
    )
    prompt = f"{system_prompt}\n<|im_start|>user\n{payload}<|im_end|>\n<|im_start|>assistant\n"
    
    response = llm(
        prompt,
        max_tokens=256,
        temperature=0.7 # Non-deterministic completion
    )
    return f"[General Agent] {response['choices'][0]['text'].strip()}"


# --- Core Orchestration Class ---
class LocalCodeOrchestrator:
    def __init__(self):
        # We define the JSON containing agent details
        self.agents_json = {
            "rag": {
                "description": "For looking up code syntax, reading files, documentation, or codebase searches in the local workspace."
            },
            "coding": {
                "description": "For writing new functions, code generation, refactoring, or creating files in the local workspace."
            },
            "general": {
                "description": "For explanations, greetings, software design chat, or answering conceptual questions."
            }
        }
        
        self.system_prompt = (
            "<|im_start|>system\n"
            "You are a routing agent. You are given a JSON containing agent details and descriptions, and a user query.\n"
            "Based on the agent descriptions, choose the most appropriate agent to handle the user's query.\n"
            "Output your decision as a valid JSON with the key 'selected_agent'.\n\n"
            f"Agent Details JSON:\n{json.dumps(self.agents_json, indent=2)}\n\n"
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
            "{\"selected_agent\": \"<agent_name>\"}<|im_end|>"
        )

    async def route_request(self, user_query: str) -> str:
        print(f"\n[Orchestrator] Processing query: '{user_query}'")
        
        prompt = f"{self.system_prompt}\n<|im_start|>user\nUser Query: {user_query}<|im_end|>\n<|im_start|>assistant\n"

        # CPU Execution Block with temperature=0.7 for non-deterministic routing decision
        try:
            response = llm(
                prompt,
                max_tokens=32,
                temperature=0.7,
                grammar=json_grammar
            )
            
            raw_json = response["choices"][0]["text"].strip()
            plan = json.loads(raw_json)
            selected_agent = plan.get("selected_agent")
            
            print(f"[Orchestrator] Route Target -> {selected_agent.upper()} (non-deterministic routing)")

            # Dispatch target agent (using user_query as payload)
            if selected_agent == "rag":
                return await run_rag_agent(user_query)
            elif selected_agent == "coding":
                return await run_coding_agent(user_query)
            
            return await run_general_agent(user_query)
        except Exception as e:
            if 'raw_json' in locals():
                print(f"[Debug Error Output] Raw JSON was:\n{raw_json}")
            return f"[Orchestrator Error] Failed to route and execute: {e}"

# --- Live Test Interface ---
async def main():
    orchestrator = LocalCodeOrchestrator()
    
    # Direct command-line argument mode
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        res = await orchestrator.route_request(query)
        print(f"\n[System Output]\n{res}\n")
        return
        
    print("=" * 60)
    print("Local CPU Semantic Orchestrator Active (Model: Llama 3.2 1B Instruct)")
    print("Type your instructions or questions. Enter 'exit' or 'quit' to stop.")
    print("=" * 60)
    
    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("Goodbye!")
                break
                
            res = await orchestrator.route_request(user_input)
            print(f"\n[System Output]\n{res}")
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[Error] {e}")

if __name__ == "__main__":
    asyncio.run(main())
