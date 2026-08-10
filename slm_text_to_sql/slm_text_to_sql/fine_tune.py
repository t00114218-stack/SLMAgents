# To connect to a TPU in Google Colab:
# 1. Go to the 'Runtime' menu at the top.
# 2. Select 'Change runtime type'.
# 3. Choose 'TPU' as the 'Hardware accelerator'.
# 4. Click 'Save'.
#
# After changing the runtime, you can run the following code to verify the TPU connection:
#
# import tensorflow as tf
# try:
#     tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
#     tf.config.experimental_connect_to_cluster(tpu_resolver)
#     tf.tpu.experimental.initialize_tpu_system(tpu_resolver)
#     strategy = tf.distribute.TPUStrategy(tpu_resolver)
#     print('TPU device:', tpu_resolver.master())
# except ValueError:
#     print('ERROR: Not connected to a TPU runtime! Please change your runtime type to TPU.')
#     print('If you are already in a TPU runtime, make sure your environment variables are set correctly.')
# except Exception as e:
#     print(f"An error occurred while connecting to TPU: {e}")
#
# If you intend to use the Qwen model for fine-tuning on TPU, note that the current setup uses PyTorch. 
# For PyTorch models on TPU, distributed training typically involves `torch_xla`. 
# The existing fine-tuning setup is configured for GPU (`torch_dtype=torch.bfloat16`, `device_map="auto"`).
# To use PyTorch on TPU, you would need to import `torch_xla` and configure the device appropriately. 
# Example for PyTorch XLA setup (uncomment if using PyTorch with TPU):
# # import torch_xla.core.xla_model as xm
# # device = xm.xla_device()
# # model.to(device) # Move model to XLA device
# # Ensure your `TrainingArguments` are compatible with XLA, or manually manage training steps with XLA.

import os
import sys
import argparse

# Helper to run package installation if requested or automatically when in interactive environments
def install_dependencies():
    import subprocess
    print("Installing packages: torch transformers datasets peft bitsandbytes trl accelerate...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch", "transformers", "datasets", "peft", "bitsandbytes", "trl", "accelerate"])

def run_fine_tuning(
    model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct",
    dataset_name="trl-lab/SQaLe-text-to-SQL",
    output_dir="./results",
    adapters_output_dir="./qwen2.5_coder_text2sql_adapters",
    merged_output_dir="./qwen2.5_coder_text2sql_merged",
    epochs=3,
    batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    max_seq_length=512,
    run_training=False,
    merge_after_training=False,
    no_quant=False
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from trl import SFTTrainer
    from transformers import TrainingArguments

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure hardware capability fallback
    if torch.cuda.is_available():
        if no_quant:
            print(f"1. Loading base model '{model_id}' on GPU without quantization (full precision)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=None,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            print(f"1. Loading base model '{model_id}' with 4-bit quantization on GPU...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
            )
    elif torch.backends.mps.is_available():
        print(f"1. [INFO] Apple Silicon GPU (MPS) detected. Loading base model '{model_id}' on MPS without quantization...")
        # MPS supports float16 to save memory
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=None,
            torch_dtype=torch.float16,
            device_map={"": "mps"},
            trust_remote_code=True,
        )
    else:
        print(f"1. [WARNING] Neither CUDA nor MPS is available. Loading base model '{model_id}' on CPU without quantization (this may take substantial RAM)...")
        # On CPU, we bypass 4-bit quantization because bitsandbytes is CUDA-only.
        # We also use float32 to avoid potential lack of CPU support for bfloat16 training operations.
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=None,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            trust_remote_code=True,
        )

    print(f"2. Loading and formatting dataset '{dataset_name}'...")
    
    def format_example(example):
        schema = example["db_schema"]
        question = example["question"]
        sql_query = example["query"]

        messages = [
            {"role": "system", "content": "You are an expert SQL assistant."},
            {"role": "user", "content": f"Schema:\n{schema}\n\nQuestion: {question}"},
            {"role": "assistant", "content": sql_query}
        ]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}

    # Correct way to load a dataset from the Hugging Face Hub when it's a specific file type within a repository
    raw_dataset = load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{dataset_name}/data/train.parquet",
        },
        split="train"
    )

    # Apply the formatting function to the dataset
    dataset = raw_dataset.map(format_example, remove_columns=raw_dataset.column_names)

    print("\nFormatted Dataset Example (first entry):")
    print(dataset[0]["text"])

    print("3. Configuring LoRA and SFTTrainer...")
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Configure training parameters based on hardware capabilities
    if torch.cuda.is_available():
        bf16_val = True
        optim_val = "paged_adamw_8bit"
        max_steps_val = -1
    elif torch.backends.mps.is_available():
        print("[INFO] Running on Apple Silicon GPU (MPS). Training will run for full epochs unless overridden.")
        bf16_val = False
        optim_val = "adamw_torch"
        max_steps_val = -1
    else:
        print("[WARNING] Running on CPU. Restricting dataset size and using max_steps=1 for verification.")
        bf16_val = False
        optim_val = "adamw_torch"
        max_steps_val = 1
        # Restrict dataset size to avoid slow mapping
        dataset = dataset.select(range(min(2, len(dataset))))

    # Training arguments
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=epochs if max_steps_val == -1 else 1,
        max_steps=max_steps_val,
        logging_steps=1,
        save_steps=50,
        save_total_limit=2,
        fp16=False,
        bf16=bf16_val,
        optim=optim_val,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
    )

    # Initialize SFTTrainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        peft_config=lora_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
    )

    # Start training
    if run_training:
        print("\nStarting training...")
        trainer.train()
        print(f"\nSaving LoRA adapters to {adapters_output_dir}...")
        trainer.save_model(adapters_output_dir)
    else:
        print("\nTraining run bypassed. Set `run_training=True` (or run with `--train`) to start fine-tuning.")
        print(f"LoRA adapters would be saved to {adapters_output_dir} after training.")

    # Merge adapters into the base model (if training was run and merge is requested)
    if run_training and merge_after_training:
        print("\nMerging adapters back into base model...")
        # Free VRAM
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Load the base model again without quantization for merging
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else {"": "cpu"},
            trust_remote_code=True,
        )

        # Load the saved PEFT model
        peft_model = PeftModel.from_pretrained(base_model, adapters_output_dir)

        # Merge LoRA weights into the base model and save
        merged_model = peft_model.merge_and_unload()
        print(f"Saving merged model to {merged_output_dir}...")
        merged_model.save_pretrained(merged_output_dir)
        tokenizer.save_pretrained(merged_output_dir)
    else:
        print(f"\nMerged model would be saved to {merged_output_dir} after merging (Code bypassed).")

    print("\nSetup process complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Qwen 2.5 Coder on SQaLe Text-to-SQL dataset using QLoRA.")
    parser.add_argument("--install", action="store_true", help="Install necessary pip dependencies first.")
    parser.add_argument("--train", action="store_true", help="Run actual training (uncomments trainer.train()).")
    parser.add_argument("--merge", action="store_true", help="Merge LoRA adapters into base model after training.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=1, help="Training batch size per device.")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-Coder-1.5B-Instruct", help="Hugging Face base model ID.")
    parser.add_argument("--dataset", type=str, default="trl-lab/SQaLe-text-to-SQL", help="Hugging Face dataset name.")
    parser.add_argument("--no-quant", action="store_true", help="Disable 4-bit quantization on GPU (use full bfloat16 instead).")
    
    args = parser.parse_args()
    
    if args.install:
        install_dependencies()
        
    run_fine_tuning(
        model_id=args.model_id,
        dataset_name=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_training=args.train,
        merge_after_training=args.merge,
        no_quant=args.no_quant
    )
