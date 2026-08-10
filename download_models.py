import os
from huggingface_hub import snapshot_download

def main():
    repo_id = "tonythethompson/Qwen2.5-1.5B-Instruct-ONNX"
    # Resolve the destination relative to this script's directory
    target_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "qwen2.5-1.5b-onnx")
    
    print(f"[System] Downloading model snapshot from Hugging Face: {repo_id}...")
    print(f"[System] Target directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=target_dir,
            ignore_patterns=["*cuda*", "*directml*"]
        )
        print("[System] Download completed successfully!")
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")

if __name__ == "__main__":
    main()
