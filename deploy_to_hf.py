import os
import sys
import shutil
from huggingface_hub import HfApi

def main():
    print("==========================================================")
    print("  Deploying SLM Agents to Hugging Face (Unified Uploader) ")
    print("==========================================================")
    print("")

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Access Token (WRITE permission): ").strip()
    if not token:
        print("❌ Error: Token cannot be empty.")
        sys.exit(1)

    api = HfApi(token=token)
    username = "spcv"
    
    # 1. Check and upload the fine-tuned SQL model if it exists
    model_dir = "models/qwen2.5_coder_text2sql_onnx"
    if os.path.exists(model_dir):
        model_repo_id = f"{username}/qwen2.5_coder_text2sql_onnx"
        print(f"\n[*] Found fine-tuned Text-to-SQL model at {model_dir}")
        print(f"[*] Creating/verifying Hugging Face model repository: {model_repo_id}...")
        try:
            api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
            print(f"[*] Uploading model weights to {model_repo_id} (this may take a few minutes)...")
            api.upload_folder(
                folder_path=model_dir,
                repo_id=model_repo_id,
                repo_type="model"
            )
            print("✅ Model uploaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not upload fine-tuned model: {e}")
    else:
        print(f"\n[*] No local fine-tuned model found at {model_dir}. Skipping model upload.")

    # 2. Build the clean Space deployment files
    temp_dir = ".hf_deploy_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("\n[*] Packing application files...")
    # Copy slm_* folders, ignoring large/unnecessary directories (.venv, git history, python cache, etc.)
    ignore_pat = shutil.ignore_patterns(".venv", ".git", ".vscode", "__pycache__", "dist", "build", "*.egg-info", "*.pyc", ".DS_Store")
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("slm_") and item != ".hf_deploy_temp":
            shutil.copytree(item, os.path.join(temp_dir, item), ignore=ignore_pat)

    # Copy website folder, ignoring git and temp files
    shutil.copytree("website", os.path.join(temp_dir, "website"), ignore=ignore_pat)

    # Remove binary files from the website copy to keep Space size lightweight
    for binary_file in ["complex_table.png", "flowchart.png", "speech.m4a"]:
        bin_path = os.path.join(temp_dir, "website", binary_file)
        if os.path.exists(bin_path):
            os.remove(bin_path)

    # Copy core configurations
    for config_file in ["Dockerfile", "requirements.txt", ".dockerignore", "main.py", "README.md"]:
        if os.path.exists(config_file):
            shutil.copy(config_file, os.path.join(temp_dir, config_file))

    # 3. Create/verify Space and upload folder
    space_repo_id = f"{username}/slm-agents"
    print(f"\n[*] Creating/verifying Hugging Face Space repository: {space_repo_id}...")
    try:
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )
        
        print(f"[*] Uploading code to Space {space_repo_id}...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=space_repo_id,
            repo_type="space"
        )
        print("\n==========================================================")
        print("🎉 Successfully deployed all agents and configurations!")
        print("==========================================================")
        print(f"Go to your Space page to watch the build progress:")
        print(f"  https://huggingface.co/spaces/{space_repo_id}")
        print("==========================================================")
    except Exception as e:
        print(f"❌ Error during Space upload: {e}")
        sys.exit(1)
    finally:
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
