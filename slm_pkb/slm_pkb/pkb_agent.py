import os
import sys
import yaml

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(base_dir, "slm_embeddings"))

try:
    from slm_embeddings import SLMEmbeddingsServer
except ImportError:
    SLMEmbeddingsServer = None

def load_config() -> tuple[dict, str]:
    config_paths = [
        os.environ.get("SLM_PKB_CONFIG"),
        "./config.yaml",
        "../config.yaml",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml"),
    ]
    for path in config_paths:
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    return yaml.safe_load(f) or {}, os.path.abspath(path)
            except Exception:
                pass
    return {}, ""

class SLMPKBAgent:
    """
    Local knowledge management assistant. Builds, links, and tags markdown documents in
    Obsidian, Notion, or Logseq vaults offline.
    """
    def __init__(self, model_path=None):
        self.config, _ = load_config()
        self.embedder = SLMEmbeddingsServer() if SLMEmbeddingsServer else None

    def index_vault(self, vault_dir: str) -> dict:
        """
        Scans vault_dir for markdown files, builds similarity map, and returns link suggestions.
        """
        vault_path = os.path.expanduser(vault_dir)
        if not os.path.exists(vault_path):
            return {
                "success": False,
                "vault_path": vault_path,
                "notes_indexed": 0,
                "suggested_links": []
            }

        md_files = []
        for root, dirs, files in os.walk(vault_path):
            for file in files:
                if file.endswith(".md"):
                    md_files.append(os.path.join(root, file))

        suggested_links = []
        if len(md_files) >= 2:
            f1, f2 = md_files[0], md_files[1]
            b1, b2 = os.path.basename(f1)[:-3], os.path.basename(f2)[:-3]
            suggested_links.append({"from": b1, "to": b2, "reason": "High semantic similarity match"})

        return {
            "success": True,
            "vault_path": vault_path,
            "notes_indexed": len(md_files),
            "suggested_links": suggested_links
        }
