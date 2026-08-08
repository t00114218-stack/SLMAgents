import os
import sys

# Ensure local path is prioritized to test package code
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_orchestrator import SLMOrchestrator

def main():
    print("[Test] Initializing SLMOrchestrator...")
    # Reuse the model in current directory
    orchestrator = SLMOrchestrator()
    
    # Custom agents list
    agents = [
        {"name": "General support agent", "description": "Handles greetings, generic chatter, and basic inquiries."},
        {"name": "Technical coding agent", "description": "Responsible for writing and editing code scripts and debugging errors."},
        {"name": "Information retrieval agent", "description": "Responsible for scanning local directories, reading files, and performing search queries."}
    ]
    
    test_cases = [
        {"q": "Hello, how are you?", "expected": "General support agent"},
        {"q": "Write a python function to compute fibonacci numbers", "expected": "Technical coding agent"},
        {"q": "Find all config files inside my workspace directory", "expected": "Information retrieval agent"}
    ]
    
    print("\n[Test] Running routing cases:")
    for idx, case in enumerate(test_cases):
        res = orchestrator.route(agents, case["q"])
        print(f"Case #{idx+1} | Query: '{case['q']}'")
        print(f"         | Expected: '{case['expected']}'")
        print(f"         | Selected: '{res}'")
        
    print("\n[Test] Dynamic routing test sequence completed.")

if __name__ == "__main__":
    main()
