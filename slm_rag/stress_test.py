import os
import sys
import time
import random

# Ensure local path is prioritized to import local slm_rag
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from slm_rag import SLMRag

# Setup data pools for generating 100 unique cases
NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", "Henry", "Ivy", "Jack"]
CITIES = ["Boston", "Seattle", "Denver", "Austin", "Miami", "Chicago", "Houston", "Dallas", "Phoenix", "Atlanta"]
PRODUCTS = ["Laptop", "Smartphone", "Tablet", "Smartwatch", "Headphones", "Camera", "Printer", "Keyboard", "Monitor", "Router"]
DATES = ["2026-01-01", "2026-02-15", "2026-03-10", "2026-04-20", "2026-05-05", "2026-06-12", "2026-07-04", "2026-08-30", "2026-09-18", "2026-10-25"]

# Large filler text block to pad context and simulate high load (stress testing)
FILLER_BLOCKS = [
    """
    SECTION 1.1 - SYSTEM OVERVIEW AND PROTOCOL ANALYSIS
    The internal network architecture relies on decentralized node coordination. Each transaction layer is authenticated
    using a zero-knowledge protocol. Security tokens are regenerated every 600 seconds to prevent replay attacks.
    System telemetry shows an average load of 45% on the main CPU clusters, with memory allocation stable at 4.2 GB.
    Backup logs are archived in sub-sector D-12 daily at 02:00 UTC. Network latency between node primary and secondary
    replicas remains under 12 milliseconds under high throughput conditions. No unauthorized intrusions were detected.
    """,
    """
    SECTION 1.2 - CRYPTOGRAPHIC KEY EXCHANGE STABILITY
    Key exchange is performed using elliptic-curve cryptography (ECC) over prime fields. Public keys are broadcasted
    to the ledger via a secure gossip protocol. Nodes validate signatures using ECDSA verification algorithms.
    If a signature mismatch occurs, the node enters a quarantine state and issues a network-wide consensus warning.
    This process is designed to run efficiently on low-power devices, using minimal RAM. The current block height
    is 784,291 with a consensus round duration of exactly 1.8 seconds.
    """,
    """
    SECTION 1.3 - DATABASE SHARDING AND STORAGE SCHEMAS
    Database tables are horizontally partitioned across 16 shards based on user ID hash values. Shard replication
    follows a master-slave configuration with synchronous replication for high-value financial tables.
    Read requests are routed to read-only replicas to minimize disk contention. Automatic vacuuming runs weekly
    to reclaim unused storage space. Indices are rebuilt during off-peak hours to optimize search latency.
    """,
    """
    SECTION 1.4 - PERIPHERAL HARDWARE CAPACITIES AND LOGISTICS
    All peripheral systems are connected via dual-channel high-speed interfaces. Optical transceivers support up
    to 100 Gbps bandwidth over single-mode fiber links. Environmental sensors monitor temperature, humidity, and
    cooling fan speeds. If chassis temperature exceeds 65 degrees Celsius, thermal throttling is initiated, and secondary
    exhaust fans are spun up to maximum capacity. Rack cabinet doors must remain closed to ensure proper airflow dynamics.
    """
]

def generate_test_cases():
    test_cases = []
    for i in range(100):
        name = NAMES[i % 10]
        city = CITIES[(i // 10) % 10]
        product = PRODUCTS[(i // 3) % 10] # Mix indexes to create unique tuples
        date = DATES[(i * 7) % 10]
        
        target_chunk = f"Log Entry ID {1000 + i}: Customer {name} of city {city} successfully purchased item {product} on purchase date {date}."
        
        # Select fillers and mix in the target chunk at a random position (needle in a haystack)
        chunks = FILLER_BLOCKS.copy()
        insert_idx = random.randint(0, len(chunks))
        chunks.insert(insert_idx, target_chunk)
        
        # Test Case 1: Ask for product name, instruct to respond in ONE WORD
        tc_product = {
            "id": i * 2,
            "chunks": chunks,
            "question": f"What specific item did {name} purchase?",
            "instruction": "Respond in EXACTLY one single word: the name of the product. Do not output any other text or punctuation.",
            "expected": product,
            "validate": lambda ans, expected=product: ans.strip().lower() == expected.lower()
        }
        
        # Test Case 2: Ask for the purchase date, instruct to respond in uppercase JSON format
        tc_date = {
            "id": i * 2 + 1,
            "chunks": chunks,
            "question": f"On what date did {name} make their purchase?",
            "instruction": "Respond only with a JSON object containing the key 'date'.",
            "expected": date,
            "validate": lambda ans, expected=date: expected in ans and '"date"' in ans
        }
        
        # Add one of the two test cases dynamically to test both questions and formats across the 100 runs
        if i % 2 == 0:
            test_cases.append(tc_product)
        else:
            test_cases.append(tc_date)
            
    return test_cases

def main():
    print("[Stress Test] Initializing SLMRag with 128k context support...")
    rag = SLMRag()
    
    test_cases = generate_test_cases()
    print(f"[Stress Test] Generated {len(test_cases)} unique stress test cases.")
    
    passed_count = 0
    total_time = 0.0
    
    print("\n" + "="*60)
    print(f"RUNNING 100 STRESS TEST CASES ON CPU")
    print("="*60)
    
    for idx, tc in enumerate(test_cases):
        # Calculate context length in characters
        ctx_len = sum(len(c) for c in tc["chunks"])
        
        start_time = time.time()
        try:
            response = rag.answer(
                chunks=tc["chunks"],
                question=tc["question"],
                instruction=tc["instruction"],
                temperature=0.0,
                max_tokens=64
            )
            duration = time.time() - start_time
            total_time += duration
            
            is_valid = tc["validate"](response)
            if is_valid:
                passed_count += 1
                result_str = "[PASS]"
            else:
                result_str = "[FAIL]"
                
            # Log every 10th run or failures to keep console neat, but track all
            if idx % 10 == 0 or not is_valid:
                print(f"Case #{idx+1:03d} | Q: '{tc['question'][:40]}...' | Dur: {duration:.2f}s | Chars: {ctx_len} | Res: '{response.replace(chr(10), ' ')}' -> {result_str}")
                
        except Exception as e:
            duration = time.time() - start_time
            total_time += duration
            print(f"Case #{idx+1:03d} | Error: {e} | Dur: {duration:.2f}s")
            
    accuracy = (passed_count / len(test_cases)) * 100
    avg_latency = total_time / len(test_cases)
    
    print("\n" + "="*60)
    print("STRESS TEST COMPLETED")
    print("="*60)
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"Passed:           {passed_count}/{len(test_cases)}")
    print(f"Overall Accuracy: {accuracy:.2f}%")
    print(f"Total Execution:  {total_time:.2f}s")
    print(f"Avg Latency/Run:  {avg_latency:.2f}s")
    print("="*60)
    
    if accuracy >= 95.0:
        print("Stress test validation successful with high accuracy!")
    else:
        print("Accuracy fell below threshold. Verify model parameters or prompt formatting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
