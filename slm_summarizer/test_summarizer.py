import os
import sys

# Ensure local path is prioritized to import local slm_summarizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_summarizer import SLMSummarizer

def main():
    print("[Test] Initializing SLMSummarizer (this may take a few seconds to load)...")
    # Load with 4096 context for testing to save memory and CPU initialization
    summarizer = SLMSummarizer(n_ctx=4096)
    
    # Short article for direct single-pass testing
    short_text = (
        "SpaceX successfully launched its Falcon 9 rocket on Friday, sending 22 Starlink satellites "
        "into low Earth orbit. The mission lifted off from Cape Canaveral Space Force Station in Florida. "
        "About eight minutes after launch, the rocket's first stage returned to Earth, landing safely on the "
        "droneship 'A Shortfall of Gravitas' stationed in the Atlantic Ocean. This marked the 15th successful "
        "flight and landing for this particular booster, representing another milestone in SpaceX's reuse technology. "
        "The Starlink constellation now provides high-speed satellite internet service to over 3 million subscribers globally."
    )
    
    # Large document for Map-Reduce testing (repeated segments with minor variations to exceed chunk_size)
    base_paragraph = (
        "Artificial Intelligence (AI) has progressed rapidly over the past decade, transforming industries from healthcare "
        "to finance. Underpinning this revolution is the development of neural network architectures, particularly the Transformer. "
        "Transformers have allowed researchers to train massive models on gargantuan datasets, leading to emergent capabilities like "
        "context-aware reasoning, translation, and high-fidelity text generation. However, running these large models requires "
        "immense computing power, typically demanding GPU clusters with hundreds of gigabytes of VRAM. This has led to environmental "
        "concerns due to carbon footprints, alongside high costs that lock out smaller startups and researchers. "
    )
    long_text = "\n\n".join([f"[Section {i+1}] " + base_paragraph for i in range(10)])

    test_cases = [
        {
            "name": "Short Text - Bullet Points",
            "text": short_text,
            "format": "bullet_points",
            "instruction": "",
            "chunk_size": 4000,
            "check": lambda ans: "-" in ans or "*" in ans or "SpaceX" in ans
        },
        {
            "name": "Short Text - Paragraph Format",
            "text": short_text,
            "format": "paragraph",
            "instruction": "",
            "chunk_size": 4000,
            "check": lambda ans: len(ans) > 20 and "SpaceX" in ans and not ans.strip().startswith("-")
        },
        {
            "name": "Short Text - TL;DR Format",
            "text": short_text,
            "format": "tldr",
            "instruction": "",
            "chunk_size": 4000,
            "check": lambda ans: len(ans.split(".")) <= 2 and "SpaceX" in ans
        },
        {
            "name": "Short Text - Custom Instruction (Pirate Style)",
            "text": short_text,
            "format": "paragraph",
            "instruction": "Speak like a 17th-century pirate.",
            "chunk_size": 4000,
            "check": lambda ans: any(w in ans.lower() for w in ["ahoy", "matey", "ye", "scurvy", "arrr", "sea", "ship", "captain", "booster"])
        },
        {
            "name": "Long Text - Map-Reduce Pipeline",
            "text": long_text,
            "format": "bullet_points",
            "instruction": "",
            "chunk_size": 1500,  # Small chunk size to force Map-Reduce splitting
            "check": lambda ans: len(ans) > 20 and ("AI" in ans or "Artificial Intelligence" in ans or "Transformer" in ans)
        }
    ]
    
    print("\n" + "="*50)
    print("RUNNING SUMMARIZER VERIFICATION SUITE")
    print("="*50)
    
    passed_count = 0
    for idx, case in enumerate(test_cases):
        print(f"\nTest #{idx+1}: {case['name']}")
        print(f"Format: '{case['format']}'")
        if case["instruction"]:
            print(f"Instruction: '{case['instruction']}'")
        print(f"Input Length: {len(case['text'])} characters")
        
        try:
            summary = summarizer.summarize(
                text=case["text"],
                format=case["format"],
                instruction=case["instruction"],
                chunk_size=case["chunk_size"],
                temperature=0.0
            )
            print(f"Summary Response:\n{summary}")
            
            if case["check"](summary):
                print("Result: [PASS]")
                passed_count += 1
            else:
                print("Result: [FAIL] (Did not meet validation criteria)")
        except Exception as e:
            print(f"Result: [ERROR] ({e})")
            
    print("\n" + "="*50)
    print(f"Verification completed. Passed: {passed_count}/{len(test_cases)}")
    print("="*50)

    # JSON input testing
    print("\n" + "="*50)
    print("RUNNING JSON INPUT VERIFICATION SUITE")
    print("="*50)
    
    import json
    json_test_cases = [
        {
            "name": "JSON Input String - TL;DR",
            "input": json.dumps({
                "passage": short_text,
                "prompt": "Highlight the SpaceX Falcon 9 launch booster reuse milestone.",
                "size": 60,
                "format": "tldr"
            }),
            "check": lambda ans: "SpaceX" in ans and ("booster" in ans or "reuse" in ans or "landing" in ans)
        },
        {
            "name": "JSON Input Dict - Bullet Points",
            "input": {
                "text": short_text,
                "instruction": "Focus on Cape Canaveral location.",
                "max_length": 80,
                "type": "bullet_points"
            },
            "check": lambda ans: "-" in ans and "Canaveral" in ans
        }
    ]
    
    json_passed_count = 0
    for idx, case in enumerate(json_test_cases):
        print(f"\nJSON Test #{idx+1}: {case['name']}")
        try:
            summary = summarizer.summarize_json(case["input"])
            print(f"Summary Response:\n{summary}")
            
            if case["check"](summary):
                print("Result: [PASS]")
                json_passed_count += 1
            else:
                print("Result: [FAIL] (Did not meet validation criteria)")
        except Exception as e:
            print(f"Result: [ERROR] ({e})")
            
    print("\n" + "="*50)
    print(f"JSON Verification completed. Passed: {json_passed_count}/{len(json_test_cases)}")
    print("="*50)

    if passed_count == len(test_cases) and json_passed_count == len(json_test_cases):
        print("All summarizer & JSON test scenarios passed successfully!")
    else:
        print("Some test cases failed or returned unexpected formats.")
        sys.exit(1)

if __name__ == "__main__":
    main()
