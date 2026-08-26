import os
import sys
import shutil
import time
from huggingface_hub import HfApi

def main():
    print("==========================================================")
    print("  Deploying SLM Agents to Hugging Face (Direct Space Deploy) ")
    print("==========================================================")

    token = os.environ.get("HF_TOKEN")
    if not token:
        token = input("Enter your Hugging Face Access Token: ").strip()
    if not token:
        print("❌ Error: Token cannot be empty.")
        sys.exit(1)

    api = HfApi(token=token)
    username = "spcv"
    space_repo_id = f"{username}/slm-agents"

    temp_dir = ".hf_deploy_temp"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    print("\n[*] Packing application codebase...")
    ignore_pat = shutil.ignore_patterns(
        ".venv", ".git", ".vscode", "__pycache__", "dist", "build", 
        "*.egg-info", "*.pyc", ".DS_Store", "*.zip", "*.onnx", "*.data", 
        "*.bin", "*.safetensors", "*.pth", "test*", "scratch*", "models*",
        "*.log", "*.workspace", "predictions*", "predict_*.json"
    )
    for item in os.listdir("."):
        if os.path.isdir(item) and item.startswith("slm_") and item != ".hf_deploy_temp":
            shutil.copytree(item, os.path.join(temp_dir, item), ignore=ignore_pat)

    # Copy website folder, ignoring binary media
    shutil.copytree("website", os.path.join(temp_dir, "website"), ignore=ignore_pat)
    for binary_file in ["complex_table.png", "flowchart.png", "speech.m4a"]:
        bin_path = os.path.join(temp_dir, "website", binary_file)
        if os.path.exists(bin_path):
            os.remove(bin_path)

    # Copy root config files
    for config_file in ["Dockerfile", "requirements.txt", ".dockerignore", "main.py", "README.md", "config.xml", "download_models.py"]:
        if os.path.exists(config_file):
            shutil.copy(config_file, os.path.join(temp_dir, config_file))

    # Calculate total size
    total_size = sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, filenames in os.walk(temp_dir) for f in filenames)
    print(f"[*] Total package size: {round(total_size / (1024 * 1024), 2)} MB")

    print(f"\n[*] Creating/verifying Hugging Face Space: {space_repo_id}...")
    try:
        api.create_repo(
            repo_id=space_repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True
        )

        print(f"[*] Uploading files to {space_repo_id}...")
        api.upload_folder(
            folder_path=temp_dir,
            repo_id=space_repo_id,
            repo_type="space",
            commit_message="feat: deploy AI chat studio, canonical SEO fixes, and agent modules"
        )
        print("\n==========================================================")
        print("🎉 Successfully deployed to Hugging Face Space!")
        print(f"🚀 Space URL: https://huggingface.co/spaces/{space_repo_id}")
        print("==========================================================")
    except Exception as e:
        print(f"❌ Error during Space upload: {e}")
        sys.exit(1)
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()
