import asyncio
import json
import time
import sys
import os

# Auto-inject virtual environment's site-packages to sys.path if running under system python
venv_site_packages = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv', 'lib', 'python3.9', 'site-packages')
if os.path.exists(venv_site_packages) and venv_site_packages not in sys.path:
    sys.path.insert(0, venv_site_packages)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slm_orchestrator import SLMOrchestrator

AGENTS = [
    {
        "name": "coding",
        "description": "For writing new functions, code generation, refactoring, or creating files in the local workspace."
    },
    {
        "name": "rag",
        "description": "For looking up code syntax, reading files, documentation, or codebase searches in the local workspace."
    },
    {
        "name": "general",
        "description": "For explanations, greetings, software design chat, or answering conceptual questions."
    }
]

TEST_CASES = [
    # 1. Standard
    {
        "query": "Write a quick Python sorting function for arrays in sort.py",
        "expected": "coding",
        "category": "Standard"
    },
    {
        "query": "Read hello.py file contents",
        "expected": "coding",
        "category": "Standard"
    },
    {
        "query": "Search codebase for database config parameters",
        "expected": "rag",
        "category": "Standard"
    },
    {
        "query": "What are the core differences between git and mercurial?",
        "expected": "general",
        "category": "Standard"
    },
    
    # 2. Cross-Keyword (Explicitly trying to confuse agent keywords)
    {
        "query": "explain the benefits of separating a RAG agent from a Coding agent",
        "expected": "general",
        "category": "Cross-Keyword"
    },
    {
        "query": "Search the codebase for the function that writes hello world",
        "expected": "rag",
        "category": "Cross-Keyword"
    },
    {
        "query": "Write a document explaining how vector search indexing works",
        "expected": "general",
        "category": "Cross-Keyword"
    },
    {
        "query": "Find where we implement code file reading in orchestrator.py",
        "expected": "rag",
        "category": "Cross-Keyword"
    },
    
    # 3. Stress Testing (Ambiguous, complex, or long inputs)
    {
        "query": "Can you check if hello.py exists and write a test file named test_hello.py if it is missing?",
        "expected": "coding",
        "category": "Stress"
    },
    {
        "query": "Where is llama-cpp-python imported in our code? Locate the exact file and explain it.",
        "expected": "rag",
        "category": "Stress"
    },
    {
        "query": "create a new folder, create files, write code inside them, and refactor existing functions",
        "expected": "coding",
        "category": "Stress"
    },
    {
        "query": "Explain how LLMs are quantized to Q4_K_M GGUF format and how LlamaGrammar constrains output schema",
        "expected": "general",
        "category": "Stress"
    },
    
    # 4. Noise & Edge Cases
    {
        "query": "Hello there! How are you today? What can you do?",
        "expected": "general",
        "category": "Noise/Edge"
    },
    {
        "query": "!!! codebase check ??? search !!! hello.py",
        "expected": "rag",
        "category": "Noise/Edge"
    },
    {
        "query": "write python",
        "expected": "coding",
        "category": "Noise/Edge"
    }
]

async def run_single_route(orchestrator, query: str) -> str:
    """Executes only the routing step of the orchestrator to verify classification."""
    return orchestrator.route(agents=AGENTS, question=query).lower().strip()

async def main():
    print("=" * 60)
    print("Local CPU Orchestrator Routing Evaluation Suite")
    print("=" * 60)
    print(f"Loading {len(TEST_CASES)} test cases across 4 categories...")
    
    orchestrator = SLMOrchestrator()
    results = []
    
    num_runs = 3 # Run each test case 3 times to evaluate non-deterministic reliability
    print(f"Running each test {num_runs} times...")
    print("-" * 60)
    
    for idx, tc in enumerate(TEST_CASES):
        print(f"[{idx+1}/{len(TEST_CASES)}] [{tc['category']}] Testing: '{tc['query'][:50]}...'")
        runs = []
        for r_idx in range(num_runs):
            try:
                selected = await run_single_route(orchestrator, tc["query"])
                runs.append(selected)
            except Exception as e:
                runs.append(f"Error ({e})")
            # Small sleep to prevent CPU throttling
            await asyncio.sleep(0.1)
            
        # Calculate success rate for this test case
        successes = sum(1 for r in runs if r == tc["expected"])
        accuracy = successes / num_runs
        
        results.append({
            "query": tc["query"],
            "expected": tc["expected"],
            "category": tc["category"],
            "runs": runs,
            "successes": successes,
            "accuracy": accuracy
        })
        print(f"      Expected: {tc['expected'].upper()} | Runs: {runs} | Success Rate: {accuracy:.0%}")
        
    print("\n" + "=" * 60)
    print("Generating Accuracy Report...")
    print("=" * 60)
    
    # Calculate category stats
    category_stats = {}
    total_runs = len(TEST_CASES) * num_runs
    total_successes = 0
    
    for res in results:
        cat = res["category"]
        if cat not in category_stats:
            category_stats[cat] = {"runs": 0, "successes": 0}
        category_stats[cat]["runs"] += num_runs
        category_stats[cat]["successes"] += res["successes"]
        total_successes += res["successes"]
        
    overall_accuracy = total_successes / total_runs
    
    # Print console summary
    print("\nCategory Breakdown:")
    for cat, stats in category_stats.items():
        cat_acc = stats["successes"] / stats["runs"]
        print(f"  - {cat:<15}: {stats['successes']}/{stats['runs']} correct ({cat_acc:.2%})")
        
    print(f"\nOverall Routing Accuracy: {total_successes}/{total_runs} ({overall_accuracy:.2%})")
    
    # Save detailed markdown report
    report_file = "accuracy_report.md"
    markdown_lines = [
        "# Routing Accuracy & Stress Test Report\n",
        f"**Date/Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n",
        f"**Model**: Qwen 2.5 1.5B Instruct GGUF (Quantized Q4_K_M)\n",
        f"**Routing Temperature**: 0.7 (Non-deterministic)\n",
        f"**Total Runs**: {total_runs} ({len(TEST_CASES)} cases x {num_runs} runs each)\n",
        "## Executive Summary\n",
        f"**Overall Accuracy**: **{overall_accuracy:.2%}** ({total_successes}/{total_runs} correct routing decisions)\n",
        "### Accuracy by Category\n",
        "| Category | Correct / Total | Accuracy % |",
        "| :--- | :---: | :---: |"
    ]
    
    for cat, stats in category_stats.items():
        cat_acc = stats["successes"] / stats["runs"]
        markdown_lines.append(f"| {cat} | {stats['successes']}/{stats['runs']} | {cat_acc:.2%} |")
        
    markdown_lines.extend([
        "\n## Detailed Test Case Results\n",
        "| # | Category | Query | Expected | Runs (Actual Decisions) | Accuracy |",
        "| :---: | :--- | :--- | :---: | :--- | :---: |"
    ])
    
    for idx, res in enumerate(results):
        runs_str = ", ".join([r.upper() for r in res["runs"]])
        markdown_lines.append(
            f"| {idx+1} | {res['category']} | `{res['query']}` | **{res['expected'].upper()}** | {runs_str} | {res['accuracy']:.0%} |"
        )
        
    # Write to local workspace
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("\n".join(markdown_lines))
        
    print(f"\n[System] Detailed accuracy report written to '{report_file}' successfully!")

if __name__ == "__main__":
    asyncio.run(main())
