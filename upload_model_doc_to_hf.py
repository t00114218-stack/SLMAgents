#!/usr/bin/env python3
import os
import sys
from huggingface_hub import HfApi, HfFolder

def upload_model_only():
    token = os.environ.get("HF_TOKEN") or HfFolder.get_token()
    if not token and sys.stdin.isatty():
        token = input("Enter your Hugging Face Access Token (WRITE permission): ").strip()
    if not token:
        print("❌ Error: HF_TOKEN environment variable or huggingface-cli token is required.")
        print("💡 Run 'export HF_TOKEN=your_token' or 'huggingface-cli login' to authenticate.")
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = "spcv/qwen2.5_coder_text2sql_onnx"
    model_dir = "models/qwen2.5_coder_text2sql_onnx"
    
    if not os.path.exists(model_dir):
        print(f"❌ Error: Model directory {model_dir} not found.")
        sys.exit(1)
        
    readme_path = os.path.join(model_dir, "README.md")
    if not os.path.exists(readme_path):
        print(f"❌ Error: {readme_path} not found.")
        sys.exit(1)

    print(f"[*] Uploading model documentation & files to Hugging Face Model: {repo_id}...")
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=model_dir,
            repo_id=repo_id,
            repo_type="model"
        )
        print("✅ Successfully updated the model repository with documentation on Hugging Face!")
        print(f"🔗 View here: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_model_only()
