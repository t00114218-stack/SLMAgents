import os
import sys

# Ensure local path is prioritized to import local slm_rag
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_rag import SLMRag

def main():
    print("[Test] Initializing SLMRag (this may take a few seconds)...")
    rag = SLMRag()
    
    # Document chunks about a fictional company called 'NebulaCorp'
    chunks = [
        "NebulaCorp was founded in 2024 by Dr. Helena Vance. It specializes in quantum-resistant encryption algorithms.",
        "The flagship product of NebulaCorp is called 'AegisShield'. It is widely used by financial organizations.",
        "In early 2026, NebulaCorp announced a major partnership with the European Space Agency to secure satellite communications."
    ]
    
    test_cases = [
        {
            "name": "Basic Fact Retrieval",
            "question": "Who founded NebulaCorp and when was it founded?",
            "instruction": "Answer the question directly and concisely.",
            "check": lambda ans: "Helena Vance" in ans and "2024" in ans
        },
        {
            "name": "Constraint Adherence (Negative Constraint)",
            "question": "What is the name of NebulaCorp's CEO in 2026?",
            "instruction": "Answer using only the provided facts. If the information is not in the text, reply exactly with: 'I don't know.'",
            "check": lambda ans: "I don't know" in ans
        },
        {
            "name": "Stylistic Adherence (Pirate Speak)",
            "question": "What is NebulaCorp's flagship product?",
            "instruction": "Answer the question, but speak like a 17th-century pirate.",
            "check": lambda ans: any(w in ans.lower() for w in ["ahoy", "matey", "ye", "shiver", "scurvy", "arrr", "sea", "treasure", "shield", "aegis", "ship", "captain"])
        },
        {
            "name": "Formatting Adherence (JSON)",
            "question": "Name the partner of NebulaCorp announced in 2026.",
            "instruction": "Respond ONLY with a valid JSON object containing the key 'partner'. Do not output any markdown formatting or prefix.",
            "check": lambda ans: '"partner"' in ans or "European Space Agency" in ans
        }
    ]
    
    print("\n" + "="*50)
    print("RUNNING RAG VERIFICATION SUITE")
    print("="*50)
    
    passed_count = 0
    for idx, case in enumerate(test_cases):
        print(f"\nTest #{idx+1}: {case['name']}")
        print(f"Question: '{case['question']}'")
        print(f"Instruction: '{case['instruction']}'")
        
        try:
            answer = rag.answer(
                chunks=chunks,
                question=case["question"],
                instruction=case["instruction"],
                temperature=0.0
            )
            print(f"Response:\n{answer}")
            
            if case["check"](answer):
                print("Result: [PASS]")
                passed_count += 1
            else:
                print("Result: [FAIL] (Did not meet validation criteria)")
        except Exception as e:
            print(f"Result: [ERROR] ({e})")
            
    print("\n" + "="*50)
    print(f"Verification completed. Passed: {passed_count}/{len(test_cases)}")
    print("="*50)
    
    if passed_count == len(test_cases):
        print("All RAG test scenarios passed successfully!")
    else:
        print("Some test cases failed or returned unexpected formats.")
        sys.exit(1)

if __name__ == "__main__":
    main()
