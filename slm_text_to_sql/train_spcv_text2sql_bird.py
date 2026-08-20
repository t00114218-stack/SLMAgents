#!/usr/bin/env python3
"""
spcv Text2SQL Fine-Tuning Pipeline for BIRD Benchmark Excellence
Model: Qwen2.5-Coder-1.5B-Instruct
Dataset: trl-lab/SQaLe-text-to-SQL-dataset (511,630 multi-table schema training pairs)
Target Hardware: Apple Silicon / CUDA GPU
"""

import os
import sys
import time
import torch

def train_spcv_model():
    print("=" * 65)
    print("🚀 SPCV TEXT2SQL FINE-TUNING PIPELINE FOR BIRD BENCHMARK")
    print("=" * 65)
    
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    
    MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    DATASET_NAME = "trl-lab/SQaLe-text-to-SQL-dataset"
    MERGED_OUTPUT = "models/qwen2.5_coder_text2sql_merged"
    ONNX_OUTPUT = "models/qwen2.5_coder_text2sql_onnx"
    
    print(f"\n[1/4] Loading Tokenizer & Base Model ({MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    print(f"✅ Device: {device} | Dtype: {dtype}")
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        device_map="auto" if device != "cpu" else None,
        trust_remote_code=True
    )
    
    print("\n[2/4] Configuring High-Rank LoRA Adapters (r=64, alpha=128)...")
    peft_config = LoraConfig(
        r=64,
        lora_alpha=128,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    print(f"\n[3/4] Loading SQaLe BIRD Dataset ({DATASET_NAME})...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    system_prompt = (
        "You are an expert SQL query writer. Follow these rules strictly:\n"
        "1. Strictly use ONLY tables and columns explicitly defined in the provided DDL schema.\n"
        "2. ALL JOIN clauses MUST come BEFORE the WHERE clause.\n"
        "3. Return ONLY the final SQL query with no explanation or markdown."
    )
    
    def format_sqale(example):
        schema = example["schema"]
        question = example["question"]
        gold_sql = example["query"]
        
        chat = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"### Database Schema\n{schema}\n\n### Question\n{question}\n\n### SQL Query"},
            {"role": "assistant", "content": gold_sql}
        ]
        text = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": text}
        
    print("Formatting 511,630 training samples for Qwen ChatML format...")
    formatted_ds = dataset.map(format_sqale, batched=False, remove_columns=dataset.column_names, num_proc=4)
    
    print("\n[4/4] Starting SFT Trainer for spcv Model...")
    training_args = SFTConfig(
        output_dir="./tmp_spcv_checkpoints",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.05,
        num_train_epochs=1,
        bf16=torch.cuda.is_available(),
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        optim="adamw_torch",
        lr_scheduler_type="cosine"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=formatted_ds,
        processing_class=tokenizer,
        args=training_args
    )
    
    print("🚀 Fine-tuning spcv Text2SQL model on SQaLe/BIRD dataset...")
    trainer.train()
    
    print(f"\nMerging LoRA adapters into {MERGED_OUTPUT}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_OUTPUT)
    tokenizer.save_pretrained(MERGED_OUTPUT)
    print("✅ Fine-tuning & merge complete!")

if __name__ == "__main__":
    train_spcv_model()
