import os
from huggingface_hub import hf_hub_download

def load_config() -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    config_paths = [
        os.environ.get("SLM_ORCHESTRATOR_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}
            except Exception as e:
                print(f"[System] Warning: Failed to load config from {path}: {e}")
    return {}

def main():
    config = load_config()
    model_config = config.get("models", {}).get("orchestrator")
    if not model_config:
        raise ValueError("models.orchestrator configuration is missing in config.yaml")
    
    config_path = model_config.get("path")
    if not config_path:
        raise ValueError("model path configuration is missing under models.orchestrator in config.yaml")
        
    config_path = os.path.expanduser(config_path)
    dest_dir = os.path.dirname(config_path) or "."
    dest_filename = os.path.basename(config_path)
        
    repo_id = model_config.get("repo_id")
    filename = model_config.get("filename")
    if not repo_id or not filename:
        raise ValueError("auto-download parameters (repo_id, filename) are missing in config.yaml")

    full_dest_path = os.path.join(dest_dir, dest_filename)
    if os.path.exists(full_dest_path):
        print(f"[System] Model file '{full_dest_path}' already exists.")
        return

    print(f"[System] Downloading {filename} from {repo_id} to {full_dest_path}...")
    try:
        if dest_dir != ".":
            os.makedirs(dest_dir, exist_ok=True)
            
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=dest_dir
        )
        
        if downloaded_path != full_dest_path and os.path.exists(downloaded_path):
            os.rename(downloaded_path, full_dest_path)
            
        print(f"[System] Download complete! Model saved to '{full_dest_path}'")
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")

if __name__ == "__main__":
    main()
