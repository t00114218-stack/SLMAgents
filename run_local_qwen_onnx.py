#!/usr/bin/env python3
"""
Local ONNX Inference Runner for Fine-Tuned Qwen (Magpie Reasoning)
Runs on macOS Apple Silicon (M1/M2/M3/M4/M5) using onnxruntime-genai.
"""

import os
import sys
import time
import onnxruntime_genai as og

def find_model_dir():
    possible_paths = [
        os.path.expanduser("~/Downloads/qwen_magpie_onnx_int4"),
        os.path.expanduser("~/Downloads/qwen_onnx_models/qwen_magpie_onnx_int4"),
        "./qwen_magpie_onnx_int4",
        "../qwen_magpie_onnx_int4",
        os.path.expanduser("~/Downloads/qwen_magpie_onnx_fp16")
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and (os.path.exists(os.path.join(path, "model.onnx")) or os.path.exists(os.path.join(path, "genai_config.json"))):
            return path
    return None

def main():
    print("=" * 65)
    print("🧠 LOCAL QWEN (0.8B) MAGPIE CLAUDE-LEVEL REASONING ENGINE")
    print("=" * 65)
    
    model_path = find_model_dir()
    if not model_path:
        print("\n❌ Could not find exported ONNX model directory!")
        print("Please make sure you unpacked qwen_full_artifacts.tar.gz into ~/Downloads/:")
        print("  cd ~/Downloads && tar -xzvf qwen_full_artifacts.tar.gz")
        sys.exit(1)
        
    print(f"\n✅ Found local ONNX model artifact at: {model_path}")
    print("⚡ Initializing ONNX Runtime GenAI Engine...")
    
    start_load = time.time()
    model = og.Model(model_path)
    tokenizer = og.Tokenizer(model)
    tokenizer_stream = tokenizer.create_stream()
    print(f"✅ Model loaded in {time.time() - start_load:.2f} seconds!")
    
    system_prompt = (
        "You are a helpful AI assistant equipped with advanced step-by-step reasoning capabilities. "
        "When given a complex logic, math, coding, or analytical problem, carefully think through the problem "
        "step-by-step inside <think>...</think> tags before providing your final answer."
    )
    
    print("\n" + "=" * 65)
    print("Type your query (or 'exit' to quit):")
    print("=" * 65)

    while True:
        try:
            user_input = input("\n👤 User > ").strip()
            if not user_input or user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{user_input}<|im_end|>\n<|im_start|>assistant\n"
            
            input_tokens = tokenizer.encode(formatted_prompt)
            
            params = og.GeneratorParams(model)
            params.set_search_options(max_length=2048, temperature=0.6, top_p=0.9)
            params.input_ids = input_tokens
            
            generator = og.Generator(model, params)
            
            print("\n🤖 Assistant > ", end="", flush=True)
            
            start_gen = time.time()
            tokens_count = 0
            
            while not generator.is_done():
                generator.compute_logits()
                generator.generate_next_token()
                
                new_token = generator.get_next_tokens()[0]
                tokens_count += 1
                token_text = tokenizer_stream.decode(new_token)
                print(token_text, end="", flush=True)
                
            elapsed = time.time() - start_gen
            tok_per_sec = tokens_count / elapsed if elapsed > 0 else 0
            print(f"\n\n⚡ [Stats: {tokens_count} tokens generated in {elapsed:.2f}s ({tok_per_sec:.1f} tok/sec)]")
            
        except KeyboardInterrupt:
            print("\nSession ended.")
            break

if __name__ == "__main__":
    main()
