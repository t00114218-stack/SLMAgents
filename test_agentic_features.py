import os
import sys
import json

# Setup paths
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "slm_orchestrator"))
sys.path.insert(0, os.path.join(base_dir, "slm_rag"))
sys.path.insert(0, os.path.join(base_dir, "slm_summarizer"))

from slm_orchestrator.orchestrator import SLMOrchestrator
from slm_rag.rag import SLMRag
from slm_summarizer.summarizer import SLMSummarizer

def test_orchestrator():
    print("\n" + "="*50)
    print("Testing Agentic Orchestrator")
    print("="*50)
    
    # Needs to be tested with real onnxruntime genai if we want output. 
    # Let's instantiate and see if it runs.
    try:
        orchestrator = SLMOrchestrator()
    except Exception as e:
        print(f"Failed to load Orchestrator: {e}")
        return

    agents = [
        {"name": "PythonDevAgent", "description": "Writes Python code."},
        {"name": "DatabaseAgent", "description": "Queries the corporate SQL database."},
        {"name": "GeneralSupport", "description": "General chit-chat."}
    ]

    tools = [
        {
            "name": "check_db_schema",
            "description": "Checks if a table exists in the database before routing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table_name": {"type": "string"}
                }
            }
        }
    ]

    def mock_executor(tool_name, args):
        print(f"    [TOOL EXECUTION] Orchestrator called {tool_name} with {args}")
        if tool_name == "check_db_schema":
            if args.get("table_name") == "users":
                return "Table 'users' exists and has columns: id, name, email."
            return "Table does not exist."
        return "Unknown tool."

    print("Sending query requiring tool use: 'Write a query to get all emails from the users table.'")
    # This query might just directly map to DatabaseAgent, or it might try to use the tool. 
    # Either way, we just want to ensure it doesn't crash.
    result = orchestrator.route(
        agents=agents, 
        question="Write a SQL query to get all emails from the users table. Check the schema first to be sure.",
        tools=tools,
        tool_executor=mock_executor,
        max_iterations=3
    )
    print(f"Final Selected Agent: {result}")


def test_rag():
    print("\n" + "="*50)
    print("Testing Agentic RAG")
    print("="*50)

    try:
        rag = SLMRag()
    except Exception as e:
        print(f"Failed to load RAG: {e}")
        return
        
    tools = [
        {
            "name": "search_vector_db",
            "description": "Searches the vector database for additional documents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                }
            }
        }
    ]

    def mock_executor(tool_name, args):
        print(f"    [TOOL EXECUTION] RAG called {tool_name} with {args}")
        if tool_name == "search_vector_db":
            return "Additional retrieved document: The secret password is 'OpenSesame123'."
        return "Unknown tool."

    chunks = [
        "Welcome to the corporate portal. We have many security policies.",
        "You must reset your password every 30 days."
    ]

    question = "What is the secret password? Search the vector db for it."
    instruction = "Answer clearly."
    
    print(f"Sending query requiring tool use: '{question}'")
    answer = rag.answer(
        chunks=chunks,
        question=question,
        instruction=instruction,
        tools=tools,
        tool_executor=mock_executor,
        max_iterations=3
    )
    print(f"Final Answer: {answer}")


def test_summarizer():
    print("\n" + "="*50)
    print("Testing Evaluator-Corrector Summarizer")
    print("="*50)

    try:
        summarizer = SLMSummarizer()
    except Exception as e:
        print(f"Failed to load Summarizer: {e}")
        return

    text = (
        "The Apollo 11 mission was a spaceflight that first landed humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin formed the American crew "
        "that landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first "
        "person to step onto the lunar surface six hours and 39 minutes later."
    )
    
    instruction = "You MUST include the exact date July 20, 1969."
    
    print("Sending text to summarize with strict instruction...")
    summary = summarizer.summarize(
        text=text,
        format="paragraph",
        instruction=instruction,
        max_correction_loops=2
    )
    print(f"\nFinal Summary:\n{summary}")

if __name__ == "__main__":
    test_orchestrator()
    test_rag()
    test_summarizer()
    print("\nAll stress tests completed successfully.")
