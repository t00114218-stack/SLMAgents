import os
import sys

# Add local package to python path
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(base_dir, "slm_code_interpreter"))

from slm_code_interpreter.code_interpreter import SLMCodeInterpreter

def main():
    print("=" * 60)
    print("SLM Code Interpreter Verification Suite")
    print("=" * 60)
    
    # 1. Initialize Code Interpreter
    print("\n--- Initializing SLMCodeInterpreter ---")
    interpreter = SLMCodeInterpreter(n_ctx=2048)
    
    instruction = "Write a python script to compute the 10th Fibonacci number and print it."
    # 2. Test code interpreter streaming
    print("\n--- Testing Streaming Execution ---")
    print(f"User Request: '{instruction}'")
    stream = interpreter.run(instruction, stream=True)
    for chunk in stream:
        print(chunk, end="", flush=True)
    print()
    
    # 3. Test code interpreter running a standard instruction
    print("\n--- Testing Run with Self-Correction Loop ---")
    print("Running with agentic self-correction loop...")
    result = interpreter.run(instruction, max_retries=3, stream=False)
    
    print("\nExecution Result:")
    print(f"Success: {result['success']}")
    print(f"Attempts: {result['attempts']}")
    print(f"Executed Code:\n{result['code']}")
    print(f"Stdout:\n{result['stdout']}")
    if result['stderr']:
         print(f"Stderr:\n{result['stderr']}")

if __name__ == "__main__":
    main()
