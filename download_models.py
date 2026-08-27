import os
from huggingface_hub import snapshot_download

# Defined HuggingFace ONNX model catalog for each agent domain (< 6GB, CPU execution)
AGENT_MODELS = [
    {
        "name": "Qwen 3.5 0.8B ONNX (Primary engine for task planner, memory, json cleaner, scraper)",
        "repo": "onnx-community/Qwen3.5-0.8B-ONNX",
        "dir": "qwen3.5-0.8b-onnx",
        "allow_patterns": [
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "chat_template.jinja",
            "onnx/decoder_model_merged_quantized.*", "onnx/embed_tokens_quantized.*"
        ]
    },
    {
        "name": "Phi 3.5 Mini INT4 AWQ ONNX (Orchestrator engine)",
        "repo": "microsoft/Phi-3.5-mini-instruct-onnx",
        "dir": "phi-3.5-mini-instruct-onnx",
        "allow_patterns": ["cpu_and_mobile/cpu-int4-awq-block-128-acc-level-4/*"]
    },
    {
        "name": "Qwen 2.5 Coder 3B ONNX (Code Interpreter engine)",
        "repo": "onnx-community/Qwen2.5-Coder-3B-Instruct",
        "dir": "qwen2.5-coder-3b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    },
    {
        "name": "Qwen 2.5 Coder Text-to-SQL ONNX (Text-to-SQL engine)",
        "repo": "spcv/qwen2.5_coder_text2sql_onnx",
        "dir": "qwen2.5_coder_text2sql_onnx",
        "allow_patterns": ["*"]
    },
    {
        "name": "Qwen 2.5 Math 1.5B ONNX (Math engine)",
        "repo": "onnx-community/Qwen2.5-Math-1.5B-Instruct",
        "dir": "qwen2.5-math-1.5b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    },
    {
        "name": "all-MiniLM-L6-v2 ONNX (Embeddings engine)",
        "repo": "onnx-community/all-MiniLM-L6-v2-ONNX",
        "dir": "all-minilm-l6-v2-onnx",
        "allow_patterns": ["*"]
    },
    {
        "name": "BGE Reranker Base ONNX (Search Orchestrator engine)",
        "repo": "onnx-community/bge-reranker-base-ONNX",
        "dir": "bge-reranker-base-onnx",
        "allow_patterns": ["*"]
    },
    {
        "name": "Whisper Small ONNX (Voice STT engine)",
        "repo": "onnx-community/whisper-small",
        "dir": "whisper-small-onnx",
        "allow_patterns": ["*"]
    },
    {
        "name": "Qwen2 VL 2B ONNX (Vision Parser engine)",
        "repo": "onnx-community/Qwen2-VL-2B-Instruct",
        "dir": "qwen2-vl-2b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    },
    {
        "name": "Llama 3.2 3B ONNX (RAG & Meeting summarization engine)",
        "repo": "onnx-community/Llama-3.2-3B-Instruct",
        "dir": "llama-3.2-3b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    },
    {
        "name": "Qwen 2.5 Coder 1.5B ONNX (CLI, Data, DB Migration, Security, Git, Web Agent engine)",
        "repo": "onnx-community/Qwen2.5-Coder-1.5B-Instruct",
        "dir": "qwen2.5-coder-1.5b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    },
    {
        "name": "Qwen 2.5 1.5B ONNX (Translation, Document Parser, Email, PDF, PKB engine)",
        "repo": "onnx-community/Qwen2.5-1.5B-Instruct",
        "dir": "qwen2.5-1.5b-onnx",
        "allow_patterns": ["*.json", "onnx/*"]
    }
]

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, "models")
    
    print("==========================================================")
    print("      SLM Agents — Hugging Face ONNX Model Downloader     ")
    print("==========================================================")
    print(f"[*] Downloading {len(AGENT_MODELS)} ONNX models (< 6GB CPU footprint)...")
    print()

    for idx, item in enumerate(AGENT_MODELS, 1):
        target_dir = os.path.join(models_dir, item["dir"])
        print(f"[{idx}/{len(AGENT_MODELS)}] {item['name']}")
        print(f"    Repo: {item['repo']} -> {target_dir}")
        os.makedirs(target_dir, exist_ok=True)
        try:
            snapshot_download(
                repo_id=item["repo"],
                local_dir=target_dir,
                allow_patterns=item["allow_patterns"]
            )
            print(f"    ✅ Successfully downloaded {item['repo']}")
        except Exception as e:
            print(f"    ⚠️ Warning: Download skipped/failed for {item['repo']}: {e}")
        print()

    print("==========================================================")
    print("🎉 All assigned agent ONNX models verified/downloaded!")
    print("==========================================================")

if __name__ == "__main__":
    main()


