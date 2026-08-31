#!/usr/bin/env python3
"""
Test General Assistant for Column Acronym Decomposition using qwen2.5_coder_text2sql_onnx.
"""
import os
import sys
import json
import time
import onnxruntime_genai as og

REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
ASSISTANT_MODEL = os.path.join(REPO_ROOT, "models", "qwen2.5_coder_text2sql_onnx")

class GeneralAssistant:
    def __init__(self, model_path=ASSISTANT_MODEL):
        print(f"Loading General Assistant from: {model_path}...")
        self.model = og.Model(model_path)
        self.tokenizer = og.Tokenizer(self.model)
        
    def expand_schema_and_plan(self, question: str, evidence: str, schema_ddl: str) -> str:
        prompt = (
            "<|im_start|>system\n"
            "You are an expert database analyst assistant. Given a database schema and user question:\n"
            "1. Expand and explain any cryptic column names, abbreviations, or acronyms in the schema that relate to the question (e.g. NumGE1500 -> 'Number scoring >= 1500', MailStreet -> 'Mailing Street', NumTstTakr -> 'Number of test takers', AvgScrRead -> 'Average Reading Score').\n"
            "2. Clarify which column should be SELECTed and which columns should be in WHERE/ORDER BY.\n"
            "Keep your output short, direct, and structured as bullet points.<|im_end|>\n"
            "<|im_start|>user\n"
            f"Schema DDL:\n{schema_ddl}\n\n"
            f"User Question:\n{question}\n"
            + (f"Evidence Hint:\n{evidence}\n" if evidence else "")
            + "<|im_end|>\n<|im_start|>assistant\n"
        )
        
        tokens = self.tokenizer.encode(prompt)
        params = og.GeneratorParams(self.model)
        params.set_search_options(max_length=len(tokens) + 200, temperature=0.0)
        generator = og.Generator(self.model, params)
        generator.append_tokens(tokens)
        
        out_tokens = []
        while not generator.is_done():
            generator.generate_next_token()
            t = generator.get_next_tokens()
            if len(t) > 0:
                out_tokens.append(int(t[0]))
        return self.tokenizer.decode(out_tokens).strip()

if __name__ == "__main__":
    assistant = GeneralAssistant()
    
    with open("leaderboard/bird_bench/data/bird_dev_500.jsonl") as f:
        samples = [json.loads(line) for line in [f.readline() for _ in range(4)]]
        
    for idx, s in enumerate(samples):
        print(f"\n{'='*60}\nSample #{idx+1}: {s['question']}")
        t0 = time.time()
        analysis = assistant.expand_schema_and_plan(s["question"], s.get("evidence", ""), s["schema_ddl"])
        elapsed = (time.time() - t0) * 1000
        print(f"General Assistant Analysis ({elapsed:.1f}ms):")
        print(analysis)
