import os
import subprocess
import urllib.request

AGENTS = [
    "slm-cli-agent", "slm-code-interpreter", "slm-data", "slm-db-migration",
    "slm-document-parser", "slm-email", "slm-embeddings", "slm-git-repo-manager",
    "slm-json-cleaner", "slm-math", "slm-meeting", "slm-memory", "slm-orchestrator",
    "slm-pdf", "slm-pkb", "slm-rag", "slm-search-orchestrator", "slm-security",
    "slm-summarizer", "slm-task-planner", "slm-text-to-sql", "slm-translation",
    "slm-vision-parser", "slm-voice", "slm-web-agent", "slm-web-scraper"
]

DIR_MAP = {
    name: name.replace("-", "_") for name in AGENTS
}
DIR_MAP["slm-data"] = "slm_data"

def get_published_packages():
    print("Checking published packages on PyPI using JSON API...")
    published = []
    for name in AGENTS:
        url = f"https://pypi.org/pypi/{name}/json"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    published.append(name)
        except Exception:
            pass
    return published

def main():
    published = get_published_packages()
    remaining = [name for name in AGENTS if name not in published]
    
    print(f"\nStatus:")
    print(f"  Published: {len(published)}/26")
    print(f"  Remaining: {len(remaining)}/26")
    
    if not remaining:
        print("\nAll packages are already published! 🎉")
        return

    print("\nPackages to publish:")
    for name in remaining:
        print(f"  - {name} (directory: {DIR_MAP[name]})")

    token = os.environ.get("TWINE_PASSWORD")
    if not token:
        token = input("\nEnter your PyPI token (starts with pypi-): ").strip()
        if not token:
            print("Token required to upload. Aborting.")
            return

    os.environ["TWINE_USERNAME"] = "__token__"
    os.environ["TWINE_PASSWORD"] = token

    print("\nStarting upload...")
    for name in remaining:
        dir_name = DIR_MAP[name]
        dist_path = f"{dir_name}/dist/*"
        print(f"\nUploading {name}...")
        try:
            subprocess.run(
                ["python3", "-m", "twine", "upload", "--skip-existing", dist_path],
                check=True
            )
            print(f"Successfully uploaded {name}!")
        except subprocess.CalledProcessError as e:
            print(f"Failed to upload {name}: {e}")
            print("Stopping due to rate limit or upload error.")
            break

if __name__ == "__main__":
    main()
