import os
from huggingface_hub import hf_hub_download

MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

def main():
    if os.path.exists(MODEL_FILE):
        print(f"[System] Model file '{MODEL_FILE}' already exists.")
        return

    print(f"[System] Downloading Qwen2.5-1.5B-Instruct-Q4_K_M.gguf from bartowski/Qwen2.5-1.5B-Instruct-GGUF...")
    try:
        downloaded_path = hf_hub_download(
            repo_id="bartowski/Qwen2.5-1.5B-Instruct-GGUF",
            filename="Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            local_dir="."
        )
        
        # Ensure it is named exactly 'qwen2.5-1.5b-instruct-q4_k_m.gguf'
        expected_filename = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
        if os.path.exists(expected_filename) and expected_filename != MODEL_FILE:
            os.rename(expected_filename, MODEL_FILE)
            
        print(f"[System] Download complete! Model saved and renamed to '{MODEL_FILE}'")
    except Exception as e:
        print(f"[Error] Failed to download model: {e}")

if __name__ == "__main__":
    main()
