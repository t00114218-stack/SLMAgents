#!/usr/bin/env python3
"""
Jarvis Labs Fast Training & ONNX Export Pipeline
Target Hardware: NVIDIA RTX PRO 6000 (96GB VRAM, 28 vCPU)
Model: Qwen 2.5/3.5 ~0.8B
Dataset: Magpie-Align/Magpie-Pro-300K-Filtered
"""

import os
import sys
import time
import shutil
import gc
import subprocess
import torch

# Prevent filling root partition on Jarvis Labs by redirecting cache to workspace directory
WORKSPACE_DIR = os.path.abspath(".")
CACHE_DIR = os.path.join(WORKSPACE_DIR, ".cache")
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["TORCH_HOME"] = os.path.join(CACHE_DIR, "torch")
os.environ["TMPDIR"] = os.path.join(CACHE_DIR, "tmp")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)
os.makedirs(os.environ["TMPDIR"], exist_ok=True)

def print_disk_usage(stage=""):
    total, used, free = shutil.disk_usage(WORKSPACE_DIR)
    print(f"💾 [Disk Usage {stage}] Free: {free / (1024**3):.2f} GB | Used: {used / (1024**3):.2f} GB | Total: {total / (1024**3):.2f} GB")

def check_environment():
    print("=" * 60)
    print("🚀 JARVIS LABS HARDWARE & ENVIRONMENT CHECK")
    print("=" * 60)
    print_disk_usage("Initial")
    if not torch.cuda.is_available():
        print("❌ CUDA GPU not detected! Make sure PyTorch with CUDA is installed.")
        sys.exit(1)
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"✅ GPU Detected: {gpu_name}")
    print(f"✅ VRAM Available: {vram_gb:.2f} GB")
    print(f"✅ PyTorch Version: {torch.__version__}")
    print("=" * 60)

def main():
    check_environment()
    
    # Imports
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    from trl import SFTConfig, SFTTrainer
    
    MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-0.8B")
    DATASET_NAME = "Magpie-Align/Magpie-Pro-300K-Filtered"
    MERGED_OUTPUT = "./qwen_magpie_merged_16bit"
    ONNX_OUTPUT_INT4 = "./qwen_magpie_onnx_int4"
    ONNX_OUTPUT_FP16 = "./qwen_magpie_onnx_fp16"
    MAX_SEQ_LEN = 2048

    print(f"\n[1/4] Loading Base Model & Tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 96GB+ VRAM allows full bfloat16 loading with large batch size and fast SDPA attention
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            trust_remote_code=True
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

    print("\n[2/4] Setting up LoRA Adapters...")
    print(f"GPU mem allocated (post-load): {torch.cuda.memory_allocated()/1e9:.2f} GB / reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")
    # Auto-discover all nn.Linear leaf module names dynamically at runtime (excluding lm_head/embeddings and grouped conv1d)
    import torch.nn as nn
    linear_leaf_names = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            leaf_name = name.split(".")[-1]
            if leaf_name not in {"lm_head", "embed_tokens", "wte", "wpe"}:
                linear_leaf_names.add(leaf_name)

    matched_target_modules = sorted(list(linear_leaf_names))
    print(f"✅ Auto-discovered ALL {len(matched_target_modules)} linear target leaf modules: {matched_target_modules}")

    standard_modules = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    delta_hits = [m for m in matched_target_modules if m not in standard_modules]
    print(f"🔬 DeltaNet / Linear-Attention specific modules matched: {delta_hits}")
    if not delta_hits:
        print("⚠️ WARNING: No linear-attention/DeltaNet modules matched candidate list — LoRA is adapting standard attention/MLP layers.")

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=matched_target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print(f"\n[3/4] Loading Dataset: {DATASET_NAME}...")
    dataset = load_dataset(DATASET_NAME, split="train")
    
    system_prompt = (
        "You are a helpful, respectful, and honest AI assistant. "
        "Provide clear, accurate, and detailed answers to the user's questions."
    )

    from transformers import DataCollatorForSeq2Seq

    response_template_ids = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def format_and_tokenize(examples):
        batch_input_ids, batch_labels = [], []
        for convs in examples["conversations"]:
            if convs and convs[0].get("from") != "system" and convs[0].get("role") != "system":
                formatted_convs = [{"role": "system", "content": system_prompt}]
            else:
                formatted_convs = []
            
            for msg in convs:
                role = msg.get("role") or ("user" if msg.get("from") == "human" else "assistant")
                content = msg.get("content") or msg.get("value") or ""
                formatted_convs.append({"role": role, "content": content})

            formatted_chat = tokenizer.apply_chat_template(
                formatted_convs,
                tokenize=False,
                add_generation_prompt=False
            )
            
            input_ids = tokenizer.encode(formatted_chat, max_length=MAX_SEQ_LEN, truncation=True, add_special_tokens=False)
            labels = list(input_ids)
            
            seq_len = len(response_template_ids)
            matched_idx = -1
            for i in range(len(input_ids) - seq_len + 1):
                if input_ids[i : i + seq_len] == response_template_ids:
                    matched_idx = i + seq_len
                    break
            if matched_idx != -1:
                for i in range(matched_idx):
                    labels[i] = -100
            else:
                # No response template found – mask the entire example to avoid training on the prompt
                labels = [-100] * len(labels)
            
            batch_input_ids.append(input_ids)
            batch_labels.append(labels)
            
        return {"input_ids": batch_input_ids, "labels": batch_labels}

    print("Pre-tokenizing & masking dataset across 16 CPU workers...")
    formatted_dataset = dataset.map(format_and_tokenize, batched=True, remove_columns=dataset.column_names, num_proc=16)

    # Count how many examples had no response template match (fully masked)
    unmatched_count = sum(1 for lbl in formatted_dataset["labels"] if all(l == -100 for l in lbl))
    print(f"⚠️ Unmatched response template count: {unmatched_count} / {len(formatted_dataset)}")

    # Verification Step: Print a sample tokenized text to verify ChatML tags before training starts
    print("\n" + "=" * 60)
    print("🔍 PRE-TRAINING BATCH CHATML TEMPLATE VERIFICATION")
    print("=" * 60)
    sample_ids = formatted_dataset[0]["input_ids"]
    sample_text = tokenizer.decode(sample_ids)
    print(sample_text[:400])
    print("...\n[Content Snippet]\n...")
    print(sample_text[-300:])
    print("=" * 60 + "\n")

    # Ensure special tokens & pad token config are assigned to prevent batch boundary corruption
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    # Use fast C++ DataCollatorForSeq2Seq for padded integer tensor batches
    collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt")

    print("\n[4/4] Fine-Tuning SFT Model on GPU...")
    training_args = SFTConfig(
        output_dir="./tmp_checkpoints",
        per_device_train_batch_size=16,  # reduced batch to fit memory
        gradient_accumulation_steps=3,  # accumulate to keep effective batch = 48
        weight_decay=0.01,
        learning_rate=3e-5,
        warmup_steps=100,
        num_train_epochs=1,
        bf16=True,
        logging_steps=20,
        save_strategy="steps",
        save_steps=1000,
        save_total_limit=2,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        dataloader_num_workers=8, # Multi-threaded background batch loading
        dataloader_pin_memory=True, # High speed host-to-GPU memory transfer
        gradient_checkpointing=True, # Enable checkpointing to save memory
        report_to="none"
    )

    try:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            args=training_args,
            max_seq_length=MAX_SEQ_LEN
        )
    except TypeError:
        trainer = SFTTrainer(
            model=model,
            train_dataset=formatted_dataset,
            processing_class=tokenizer,
            data_collator=collator,
            args=training_args
        )

    # 🧪 Immediate Code Debugging Diagnostic: Verify loss mask & Prompt Masking Ratio % across entire batch
    print("\n" + "=" * 60)
    print("🧪 RUNNING LOSS MASK DIAGNOSTIC TEST ON FIRST TRAIN BATCH (ALL SAMPLES)")
    print("=" * 60)
    trainer_batch = next(iter(trainer.get_train_dataloader()))
    all_labels = trainer_batch["labels"].cpu().numpy()
    
    sample_ratios = []
    for idx, label_seq in enumerate(all_labels):
        masked = (label_seq == -100).sum()
        tot = len(label_seq)
        ratio = (masked / tot) * 100 if tot > 0 else 0
        sample_ratios.append(ratio)
        print(f"  - Sample #{idx+1}: Masked {masked}/{tot} tokens ({ratio:.2f}% Prompt Masking)")

    avg_mask_ratio = sum(sample_ratios) / len(sample_ratios)
    print(f"DEBUG: Average Prompt Masking Ratio across batch: {avg_mask_ratio:.2f}%")
    if min(sample_ratios) == 0:
        raise RuntimeError("❌ ERROR: At least one sample in batch failed response template matching! Loss is leaking into prompts.")
    else:
        print(f"✅ SUCCESS: All batch samples correctly masked assistant prompts (Avg: {avg_mask_ratio:.2f}%).")
    print("=" * 60 + "\n")
    print(f"GPU mem after first batch: {torch.cuda.memory_allocated()/1e9:.2f} GB / reserved: {torch.cuda.memory_reserved()/1e9:.2f} GB")

    DRY_RUN = os.environ.get("DRY_RUN", "0").lower() in ("1", "true", "yes") or "--dry-run" in sys.argv
    if DRY_RUN:
        print("=" * 60)
        print("🚀 DRY RUN COMPLETED SUCCESSFULLY!")
        print("✅ Model loaded, module targets matched, LoRA parameters computed, and batch masking verified cleanly.")
        print("=" * 60)
        return

    start_train_time = time.time()
    last_checkpoint = None
    if os.path.exists("./tmp_checkpoints"):
        checkpoints = [os.path.join("./tmp_checkpoints", d) for d in os.listdir("./tmp_checkpoints") if d.startswith("checkpoint-")]
        if checkpoints:
            last_checkpoint = max(checkpoints, key=os.path.getmtime)
            print(f"🔄 Found existing checkpoint! Resuming training from {last_checkpoint}...")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    print(f"\n✅ Training completed in {(time.time() - start_train_time) / 60:.2f} minutes!")

    # Merge LoRA
    print(f"\nMerging LoRA adapters into 16-bit model -> {MERGED_OUTPUT}...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(MERGED_OUTPUT)
    tokenizer.save_pretrained(MERGED_OUTPUT)
    print("✅ Model merged successfully!")

    # Export to ONNX via onnxruntime-genai builder
    print("\n" + "=" * 60)
    print("⚡ EXPORTING TO ONNX RUNTIME GENAI FORMAT")
    print("=" * 60)
    
    # Ensure builder dependencies (onnx, onnxscript) are installed
    print("Ensuring ONNX builder dependencies (onnx, onnxscript)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "onnx", "onnxscript"], check=False)
    
    cmd_int4 = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", MERGED_OUTPUT,
        "-o", ONNX_OUTPUT_INT4,
        "-p", "int4",
        "-e", "cpu"
    ]
    print(f"Running ONNX INT4 Export: {' '.join(cmd_int4)}")
    subprocess.run(cmd_int4, check=True)
    print(f"✅ ONNX INT4 model saved to {ONNX_OUTPUT_INT4}")

    cmd_fp16 = [
        sys.executable, "-m", "onnxruntime_genai.models.builder",
        "-m", MERGED_OUTPUT,
        "-o", ONNX_OUTPUT_FP16,
        "-p", "fp16",
        "-e", "cuda"
    ]
    print(f"\nRunning ONNX FP16 Export: {' '.join(cmd_fp16)}")
    subprocess.run(cmd_fp16, check=True)
    print(f"✅ ONNX FP16 model saved to {ONNX_OUTPUT_FP16}")

    # Cleanup temporary checkpoints to save disk space
    print("\n🧹 Cleaning up temporary files to conserve disk space...")
    if os.path.exists("./tmp_checkpoints"):
        shutil.rmtree("./tmp_checkpoints", ignore_errors=True)
        print("  - Removed ./tmp_checkpoints")
    
    # Compress ONNX outputs into single archive for easy download
    print("\n📦 Compressing ONNX models into qwen_onnx_models.tar.gz for easy download...")
    subprocess.run(["tar", "-czvf", "qwen_onnx_models.tar.gz", ONNX_OUTPUT_INT4, ONNX_OUTPUT_FP16], check=False)
    print("✅ Created qwen_onnx_models.tar.gz!")

    print_disk_usage("Final")
    print("\n🎉 ALL DONE! Your ONNX models are ready for deployment!")

if __name__ == "__main__":
    main()
