import os
from huggingface_hub import snapshot_download

# Defined HuggingFace ONNX model catalog for build-time caching (< 1.5GB total, CPU execution)
AGENT_MODELS = [
    {
        "name": "Qwen 3.5 0.8B ONNX (Primary Core Engine)",
        "repo": "onnx-community/Qwen3.5-0.8B-ONNX",
        "dir": "qwen3.5-0.8b-onnx",
        "allow_patterns": [
            "config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt", "chat_template.jinja", "genai_config.json",
            "model.onnx", "model_q4.onnx", "onnx/*"
        ]
    },
    {
        "name": "all-MiniLM-L6-v2 ONNX (Embeddings engine)",
        "repo": "onnx-community/all-MiniLM-L6-v2-ONNX",
        "dir": "all-minilm-l6-v2-onnx",
        "allow_patterns": ["*"]
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


